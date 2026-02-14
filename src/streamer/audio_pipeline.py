import logging
import time
from collections import deque
from threading import Lock
from typing import Any, Iterator

import numpy as np

from streamer.track_decoders.track_decoder import (
    TrackFrame,
    TrackDecoder,
    AudioTrackDecoderConstructor,
    TrackDecodeError,
    TrackEOFError,
    TrackOpenError,
)
from streamer.sinks.sink import (
    AudioSinkConstructor,
)
from streamer.encoder_senders.encoder_sender import (
    EncoderSender,
    EncoderSenderConstructor,
)
from streamer.get_next_track_from_rainwave import (
    GetNextTrackFromRainwaveBlockingFn,
    MarkTrackInvalidOnRainwaveFireAndForgetFn,
)
from streamer.stream_config import ShouldStopFn, StreamConfig, sample_rate, channels
import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst

ahead_buffer_ms = 200
ahead_buffer_seconds = ahead_buffer_ms / 1000


class AudioPipelineGracefulShutdownError(Exception):
    pass


class AudioPipeline:
    def __init__(
        self,
        config: StreamConfig,
        server_connector: AudioSinkConstructor,
        encoder_sender: EncoderSenderConstructor,
        audio_track: AudioTrackDecoderConstructor,
        get_next_track_from_rainwave: GetNextTrackFromRainwaveBlockingFn,
        mark_track_invalid_on_rainwave: MarkTrackInvalidOnRainwaveFireAndForgetFn,
        should_stop: ShouldStopFn,
        use_realtime_wait: bool,
        show_performance: bool,
    ) -> None:
        self._config = config
        self._realtime_start = time.monotonic()
        self._realtime_samples_sent = 0
        self._encoders: list[EncoderSender] = []
        self._get_next_track_from_rainwave = get_next_track_from_rainwave
        self._mark_track_invalid_on_rainwave = mark_track_invalid_on_rainwave
        self._should_stop = should_stop
        self._close_lock = Lock()
        self._closed = False
        self._audio_track = audio_track
        self._use_realtime_wait = use_realtime_wait
        self._show_performance = show_performance
        self.track_change_counter = 0

        try:
            self._encoders.append(
                encoder_sender(config, "mp3", server_connector, should_stop)
            )
            self._encoders.append(
                encoder_sender(config, "ogg", server_connector, should_stop)
            )
        except Exception:
            for encoder in reversed(self._encoders):
                encoder.close()
            raise

    def _encode_frame(self, frame: TrackFrame, realtime_wait: bool) -> None:
        self._raise_if_shutting_down()

        if frame.samples <= 0:
            return
        if self._use_realtime_wait and realtime_wait:
            self._realtime_wait(frame.samples)

        for encoder in self._encoders:
            encoder.encode_and_send(frame.buffer)

    def _raise_if_shutting_down(self) -> None:
        if self._should_stop():
            raise AudioPipelineGracefulShutdownError()

    def _handle_invalid_next_track(self, exc: Exception) -> None:
        if isinstance(exc, (TrackDecodeError, TrackOpenError)):
            logging.error(
                "Decode failed for track %s; marking invalid",
                exc.path,
                exc_info=exc,
            )
            self._mark_track_invalid_on_rainwave(exc.path)
        else:
            logging.error("Attempt to get a track from Rainwave failed", exc_info=exc)
            # If we don't know what error occurred, it should be re-thrown up the stack
            # to fail fast.
            raise

    def _get_next_track(
        self, get_start_buffer: bool = True
    ) -> tuple[TrackDecoder, deque[TrackFrame]]:
        # Attempt to fetch from Rainwave's backend continually until Rainwave responds.
        while True:
            self._raise_if_shutting_down()

            try:
                next_track_info = self._get_next_track_from_rainwave()
                next_track = self._audio_track(next_track_info)
                try:
                    next_track_start_buffer: deque[TrackFrame] = (
                        next_track.get_start_buffer() if get_start_buffer else deque()
                    )
                except Exception:
                    next_track.close()
                    raise
                logging.info(f"Next song queued: {next_track_info.path}")
                return (next_track, next_track_start_buffer)
            except (TrackDecodeError, TrackOpenError) as e:
                self._handle_invalid_next_track(e)
                # Wait 2 seconds before trying to fetch again from Rainwave's backend
                self._wait_for_retry_or_shutdown(2.0)
            # On any other exception fail-fast.

    def stream_tracks(
        self,
    ) -> None:
        current_track: TrackDecoder | None = None
        perf_timer = time.monotonic()
        try:
            (current_track, _) = self._get_next_track(get_start_buffer=False)

            while True:
                self._realtime_start = time.monotonic()
                self._realtime_samples_sent = 0

                try:
                    for frame in current_track.get_frames():
                        self._encode_frame(frame, realtime_wait=True)
                except TrackEOFError:
                    pass
                except TrackDecodeError as e:
                    self._handle_invalid_next_track(e)
                    # Clearing the audio buffer here will cause the crossfade check to be skipped
                    # since the resulting frames deque will be 0-length.
                    current_track.audio_buffer.clear()
                    current_track.audio_buffer_samples = 0
                # On any other error, fail fast.

                (next_track, next_track_start) = self._get_next_track()

                do_crossfade = self._detect_fade(
                    current_track.audio_buffer, direction="out"
                ) and self._detect_fade(next_track_start, direction="in")
                if do_crossfade:
                    for frame in self._mix_crossfade(
                        current_track.audio_buffer, next_track_start
                    ):
                        self._encode_frame(frame, realtime_wait=True)
                else:
                    for frame in current_track.audio_buffer:
                        self._encode_frame(frame, realtime_wait=True)
                    for frame in next_track_start:
                        self._encode_frame(frame, realtime_wait=True)

                finished_track = current_track
                current_track = next_track
                finished_track.close()

                if self._show_performance and self.track_change_counter > 100:
                    print("Perf: %.2f" % (time.monotonic() - perf_timer))
                    # I only need to see 1 sample for perf testing of the process.
                    # Anything after the first will just be noise.
                    self._show_performance = False

                self.track_change_counter += 1

        finally:
            if current_track is not None:
                current_track.close()
            self.close()

    def close(self) -> None:
        with self._close_lock:
            if self._closed:
                return
            self._closed = True

        for encoder in self._encoders:
            encoder.close()

    #########################################################################
    # Beware: below is AI slop territory
    #########################################################################

    def _buffer_to_float32(self, buffer: Any) -> np.ndarray:
        success, map_info = buffer.map(Gst.MapFlags.READ)
        if not success:
            return np.empty(0, dtype=np.float32)
        try:
            return np.frombuffer(map_info.data, dtype=np.float32).copy()
        finally:
            buffer.unmap(map_info)

    def _slice_frame(
        self, frame: TrackFrame, start_sample: int, sample_count: int
    ) -> TrackFrame:
        if sample_count <= 0:
            empty = Gst.Buffer.new_allocate(None, 0, None)
            if empty is None:
                raise RuntimeError("Failed to allocate empty Gst.Buffer.")
            return TrackFrame(buffer=empty, samples=0)

        bytes_per_sample = channels * 4
        start_byte = start_sample * bytes_per_sample
        length_bytes = sample_count * bytes_per_sample
        success, map_info = frame.buffer.map(Gst.MapFlags.READ)
        if not success:
            raise RuntimeError("Failed to map Gst.Buffer for slicing.")
        try:
            new_buffer = Gst.Buffer.new_allocate(None, length_bytes, None)
            if new_buffer is None:
                raise RuntimeError("Failed to allocate Gst.Buffer slice.")
            new_buffer.fill(0, map_info.data[start_byte : start_byte + length_bytes])
            return TrackFrame(buffer=new_buffer, samples=sample_count)
        finally:
            frame.buffer.unmap(map_info)

    def _detect_fade(self, frames: deque[TrackFrame], *, direction: str) -> bool:
        if not frames:
            return False

        total_samples = sum(frame.samples for frame in frames)
        if total_samples < (sample_rate * 0.5):  # Need at least 0.5s to judge fade
            return False

        window = max(1, int(total_samples / 20))
        rms: list[float] = []
        window_sum = 0.0
        window_samples = 0

        for frame in frames:
            frame_samples = frame.samples
            if frame_samples == 0:
                continue
            interleaved = self._buffer_to_float32(frame.buffer)
            if interleaved.size <= 0:
                continue
            samples = interleaved.reshape(frame_samples, channels)
            start = 0
            while start < frame_samples:
                take = min(window - window_samples, frame_samples - start)
                chunk = samples[start : start + take]
                window_sum += float(np.sum(chunk * chunk))
                window_samples += take
                start += take

                if window_samples >= window:
                    rms.append(float(np.sqrt(window_sum / (window_samples * channels))))
                    window_sum = 0.0
                    window_samples = 0

        if window_samples > 0:
            rms.append(float(np.sqrt(window_sum / (window_samples * channels))))

        if len(rms) < 2:
            return False

        slope = np.polyfit(np.arange(len(rms)), rms, 1)[0]
        threshold = 1e-5

        if direction == "out":
            return slope < -threshold
        return slope > threshold

    def _mix_crossfade(
        self, tail: deque[TrackFrame], head: deque[TrackFrame]
    ) -> Iterator[TrackFrame]:
        if not tail or not head:
            return

        tail_samples = sum(frame.samples for frame in tail)
        if tail_samples <= 0:
            return

        head_samples = sum(frame.samples for frame in head)
        if head_samples <= 0:
            return

        # Get the smallest size of the buffers
        max_fade_length_in_samples = min(tail_samples, head_samples)

        # Shrink the tail buffer so we only need to handle the "tail is shorter/equal" case.
        while tail_samples > max_fade_length_in_samples:
            tail_frame = tail.popleft()
            tail_samples -= tail_frame.samples
            yield tail_frame
        if tail_samples <= 0:
            while head:
                yield head.popleft()
            return

        fade_denom = tail_samples - 1
        mixed_samples = 0

        def _take_head_samples(count: int) -> np.ndarray:
            out = np.zeros((count, channels), dtype=np.float32)
            copied = 0
            while copied < count and head:
                head_frame = head.popleft()
                frame_samples = head_frame.samples
                if frame_samples == 0:
                    continue
                frame_data = self._buffer_to_float32(head_frame.buffer)
                if frame_data.size <= 0:
                    continue
                frame_data = frame_data.reshape(frame_samples, channels)

                take = min(frame_samples, count - copied)
                out[copied : copied + take] = frame_data[:take]
                copied += take

                if take < frame_samples:
                    head.appendleft(
                        self._slice_frame(
                            head_frame,
                            start_sample=take,
                            sample_count=frame_samples - take,
                        )
                    )

            return out

        while tail:
            tail_frame = tail.popleft()
            frame_samples = tail_frame.samples
            if frame_samples == 0:
                continue

            head_frame = _take_head_samples(frame_samples)
            if fade_denom <= 0:
                fade_in = np.ones((frame_samples, 1), dtype=np.float32)
            else:
                fade_in = np.linspace(
                    mixed_samples / fade_denom,
                    (mixed_samples + frame_samples - 1) / fade_denom,
                    num=frame_samples,
                    dtype=np.float32,
                ).reshape(frame_samples, 1)
            fade_out = 1.0 - fade_in
            mixed_samples += frame_samples

            tail_data = self._buffer_to_float32(tail_frame.buffer)
            if tail_data.size <= 0:
                continue
            tail_data = tail_data.reshape(frame_samples, channels)
            np.multiply(tail_data, fade_out, out=tail_data, casting="unsafe")
            np.multiply(head_frame, fade_in, out=head_frame, casting="unsafe")
            mixed = tail_data + head_frame

            out_buffer = Gst.Buffer.new_allocate(None, mixed.nbytes, None)
            if out_buffer is None:
                raise RuntimeError("Failed to allocate mixed Gst.Buffer.")
            out_buffer.fill(0, mixed.astype(np.float32, copy=False).tobytes())
            yield TrackFrame(buffer=out_buffer, samples=frame_samples)

        # Flush any head frames that were not part of the overlap.
        while head:
            yield head.popleft()

    def _realtime_wait(self, samples_count: int) -> None:
        if samples_count <= 0:
            return
        self._realtime_samples_sent += samples_count
        target_elapsed = self._realtime_samples_sent / sample_rate
        elapsed = time.monotonic() - self._realtime_start
        sleep_time = target_elapsed - elapsed - ahead_buffer_seconds
        if sleep_time > 0:
            time.sleep(sleep_time)

    def _wait_for_retry_or_shutdown(self, seconds: float) -> None:
        retry_deadline = time.monotonic() + seconds
        while True:
            self._raise_if_shutting_down()
            remaining = retry_deadline - time.monotonic()
            if remaining <= 0:
                return
            time.sleep(min(0.2, remaining))
