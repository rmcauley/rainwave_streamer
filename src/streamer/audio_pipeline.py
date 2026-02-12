#########################################################################
# This file was human-written, until the AI slop warning mid-way.
#########################################################################

import logging
import time
from collections import deque
from typing import Callable, Iterator

import av
import numpy as np

from streamer.audio_track import AudioTrack, AudioTrackEOFError, AudioTrackInfo
from streamer.stream_constants import sample_rate, channels, layout
from streamer.encoder import Encoder
from streamer.icecast_connection import IcecastConnection
from streamer.stream_config import StreamConfig

ahead_buffer_ms = 200
ahead_buffer_seconds = ahead_buffer_ms / 1000


class AudioPipeline:
    _config: StreamConfig
    _realtime_start: float
    _realtime_samples_sent: int

    def __init__(self, config: StreamConfig) -> None:
        self._config = config
        self._realtime_start: float = time.monotonic()
        self._realtime_samples_sent = 0

        mp3_conn = IcecastConnection(config, config.mp3, fmt="mp3")
        opus_conn = IcecastConnection(config, config.opus, fmt="ogg")

        self._encoders = (
            Encoder(
                mp3_conn,
                codec_name="mp3",
                fmt="mp3",
            ),
            Encoder(
                opus_conn,
                codec_name="opus",
                fmt="ogg",
            ),
        )

    def _encode_frame(self, np_frame: np.ndarray, realtime_wait: bool) -> None:
        if np_frame.shape[1] == 0:
            return
        if realtime_wait:
            self._realtime_wait(np_frame.shape[1])

        av_frame = av.AudioFrame.from_ndarray(
            np_frame.astype(np.float32, copy=False),
            format="fltp",
            layout=layout,
        )
        av_frame.sample_rate = sample_rate
        for encoder in self._encoders:
            encoder.encode(av_frame)

    def _get_next_track(
        self, next_track_blocking: Callable[[], AudioTrackInfo]
    ) -> AudioTrack:
        next_track_info = next_track_blocking()
        logging.info(f"Next song queued: {next_track_info.path}")
        return AudioTrack(next_track_info)

    def stream_tracks(
        self,
        next_track_blocking: Callable[[], AudioTrackInfo],
        should_stop: Callable[[], bool] = lambda: False,
    ) -> None:
        current_track = self._get_next_track(next_track_blocking)

        while current_track:
            if should_stop():
                return

            self._realtime_start = time.monotonic()
            self._realtime_samples_sent = 0

            try:
                for frame in current_track.get_frames():
                    if should_stop():
                        return
                    self._encode_frame(frame, realtime_wait=True)
            except AudioTrackEOFError:
                pass

            if should_stop():
                return

            next_track = self._get_next_track(next_track_blocking)
            next_track_start = next_track.get_start_crossfade_buffer()

            fade_out_smart = self._detect_fade(
                current_track.audio_buffer, direction="out"
            )
            fade_in_smart = self._detect_fade(next_track_start, direction="in")
            if fade_in_smart and fade_out_smart:
                for frame in self._mix_crossfade(
                    current_track.audio_buffer, next_track_start
                ):
                    if should_stop():
                        return
                    self._encode_frame(frame, realtime_wait=True)
            else:
                for frame in current_track.audio_buffer:
                    if should_stop():
                        return
                    self._encode_frame(frame, realtime_wait=True)
                for frame in next_track_start:
                    if should_stop():
                        return
                    self._encode_frame(frame, realtime_wait=True)

            current_track = next_track

    def close(self) -> None:
        for encoder in self._encoders:
            encoder.close()

    #########################################################################
    # Beware: below is AI slop territory
    #########################################################################

    def _detect_fade(self, frames: deque[np.ndarray], *, direction: str) -> bool:
        if not frames:
            return False

        total_samples = sum(frame.shape[1] for frame in frames)
        if total_samples < (sample_rate * 0.5):  # Need at least 0.5s to judge fade
            return False

        window = max(1, int(total_samples / 20))
        rms: list[float] = []
        window_sum = 0.0
        window_samples = 0

        for frame in frames:
            frame_samples = frame.shape[1]
            if frame_samples == 0:
                continue
            start = 0
            while start < frame_samples:
                take = min(window - window_samples, frame_samples - start)
                chunk = frame[:, start : start + take]
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
        self, tail: deque[np.ndarray], head: deque[np.ndarray]
    ) -> Iterator[np.ndarray]:
        if not tail or not head:
            return

        tail_samples = sum(frame.shape[1] for frame in tail)
        if tail_samples <= 0:
            return

        head_samples = sum(frame.shape[1] for frame in head)
        if head_samples <= 0:
            return

        # Get the smallest size of the buffers
        max_fade_length_in_samples = min(tail_samples, head_samples)

        # Shrink the tail buffer so we only need to handle the "tail is shorter/equal" case.
        while tail_samples > max_fade_length_in_samples:
            tail_frame = tail.popleft()
            tail_samples -= tail_frame.shape[1]
            yield tail_frame
        if tail_samples <= 0:
            while head:
                yield head.popleft()
            return

        fade_denom = tail_samples - 1
        mixed_samples = 0

        def _take_head_samples(count: int) -> np.ndarray:
            out = np.zeros((channels, count), dtype=np.float32)
            copied = 0
            while copied < count and head:
                head_frame = head.popleft()
                frame_samples = head_frame.shape[1]
                if frame_samples == 0:
                    continue

                take = min(frame_samples, count - copied)
                out[:, copied : copied + take] = head_frame[:, :take]
                copied += take

                if take < frame_samples:
                    head.appendleft(head_frame[:, take:])

            return out

        while tail:
            tail_frame = tail.popleft()
            frame_samples = tail_frame.shape[1]
            if frame_samples == 0:
                continue

            head_frame = _take_head_samples(frame_samples)
            if fade_denom <= 0:
                fade_in = np.ones(frame_samples, dtype=np.float32)
            else:
                fade_in = np.linspace(
                    mixed_samples / fade_denom,
                    (mixed_samples + frame_samples - 1) / fade_denom,
                    num=frame_samples,
                    dtype=np.float32,
                )
            fade_out = 1.0 - fade_in
            mixed_samples += frame_samples
            yield tail_frame * fade_out + head_frame * fade_in

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
