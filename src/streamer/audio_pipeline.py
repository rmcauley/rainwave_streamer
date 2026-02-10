import logging
import math
from typing import Callable, Iterator
import itertools
import time
from dataclasses import dataclass
from collections import deque

import av
from av.audio.resampler import AudioResampler
import numpy as np

from streamer.stream_constants import sample_rate, channels, layout
from streamer.encoder import Encoder
from streamer.icecast_connection import IcecastConnection
from streamer.stream_config import StreamConfig
from streamer.track_info import TrackInfo

crossfade_seconds = 5
buffer_seconds = 2
ahead_buffer_ms_default = 200


class AudioPipeline:
    def __init__(
        self, config: StreamConfig, *, ahead_buffer_ms: int = ahead_buffer_ms_default
    ) -> None:
        self._config = config
        self._buffer_samples = int(sample_rate * max(crossfade_seconds, buffer_seconds))
        self._queue: deque[np.ndarray] = deque()
        self._queue_samples: int = 0
        self._realtime_start: float | None = None
        self._realtime_samples_sent = 0
        self._ahead_buffer_seconds = max(0.0, ahead_buffer_ms / 1000.0)

        self._resampler = AudioResampler(
            format="fltp",
            layout=layout,
            rate=sample_rate,
        )

        mp3_conn = IcecastConnection(config, config.mp3, fmt="mp3", allow_metadata=True)
        opus_conn = IcecastConnection(
            config, config.opus, fmt="ogg", allow_metadata=False
        )

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
        self._mp3_conn = mp3_conn

    def close(self) -> None:
        for encoder in self._encoders:
            encoder.close()
        self._realtime_start = None
        self._realtime_samples_sent = 0
        self._clear_queue()

    def _enqueue_samples(self, samples: np.ndarray) -> Iterator[np.ndarray]:
        if samples.shape[1] == 0:
            return iter(())
        self._queue.append(samples)
        self._queue_samples += samples.shape[1]
        return self._dequeue_ready()

    def _dequeue_ready(self) -> Iterator[np.ndarray]:
        ready: list[np.ndarray] = []
        while self._queue_samples > self._buffer_samples:
            excess = self._queue_samples - self._buffer_samples
            head = self._queue[0]
            if head.shape[1] <= excess:
                ready.append(head)
                self._queue.popleft()
                self._queue_samples -= head.shape[1]
            else:
                emit = head[:, :excess]
                keep = head[:, excess:]
                ready.append(emit)
                self._queue[0] = keep
                self._queue_samples -= excess
        return iter(ready)

    def _peek_queue(self) -> np.ndarray:
        if not self._queue:
            return np.zeros((channels, 0), dtype=np.float32)
        if len(self._queue) == 1:
            return self._queue[0]
        return np.concatenate(list(self._queue), axis=1)

    def _clear_queue(self) -> None:
        self._queue.clear()
        self._queue_samples = 0

    def _trim_silence(
        self, samples: np.ndarray, threshold_db: float = -60.0
    ) -> np.ndarray:
        """
        Removes trailing silence from the end of the numpy array.
        """
        if samples.shape[1] == 0:
            return samples

        # Convert dB threshold to linear amplitude
        threshold_linear = math.pow(10, threshold_db / 20)

        # Check absolute amplitude across all channels (max projection)
        # We look for the last index where amplitude > threshold
        amplitudes = np.max(np.abs(samples), axis=0)

        # Find indices where audio is audible
        audible_indices = np.where(amplitudes > threshold_linear)[0]

        if len(audible_indices) == 0:
            # The entire buffer is silence
            return np.zeros((samples.shape[0], 0), dtype=np.float32)

        # The new end is the last audible sample + 1
        last_audible = audible_indices[-1] + 1

        if last_audible < samples.shape[1]:
            trimmed = samples.shape[1] - last_audible
            logging.debug(
                f"Trimmed {trimmed} silent samples ({trimmed/sample_rate:.2f}s) from buffer."
            )
            return samples[:, :last_audible]

        return samples

    def _detect_fade(self, samples: np.ndarray, *, direction: str) -> bool:
        if samples.shape[1] < (sample_rate * 0.5):  # Need at least 0.5s to judge fade
            return False

        # Analyze RMS energy
        window = max(1, int(samples.shape[1] / 20))
        rms: list[float] = []
        for i in range(0, samples.shape[1], window):
            chunk = samples[:, i : i + window]
            if chunk.shape[1] == 0:
                continue
            rms.append(float(np.sqrt(np.mean(chunk * chunk))))

        if len(rms) < 2:
            return False

        slope = np.polyfit(np.arange(len(rms)), rms, 1)[0]
        threshold = 1e-5

        if direction == "out":
            return slope < -threshold
        else:
            return slope > threshold

    def _mix_crossfade(self, tail: np.ndarray, head: np.ndarray) -> np.ndarray:
        length = min(tail.shape[1], head.shape[1])
        if length == 0:
            return np.zeros((channels, 0), dtype=np.float32)

        fade_out = np.linspace(1.0, 0.0, num=length, dtype=np.float32)
        fade_in = np.linspace(0.0, 1.0, num=length, dtype=np.float32)

        mixed = tail[:, -length:] * fade_out + head[:, :length] * fade_in
        return mixed

    def _encode_samples_scaled(self, samples: np.ndarray) -> None:
        if samples.shape[1] == 0:
            return
        self._realtime_wait(samples.shape[1])
        samples = np.clip(samples, -1.0, 1.0)
        frame = av.AudioFrame.from_ndarray(
            samples.astype(np.float32, copy=False),
            format="fltp",
            layout=layout,
        )
        frame.sample_rate = sample_rate
        for encoder in self._encoders:
            encoder.encode(frame)

    def _encode_samples(self, samples: np.ndarray, *, gain_db: float) -> None:
        gain = math.pow(10.0, gain_db / 20.0)
        self._encode_samples_scaled(samples * gain)

    def _realtime_wait(self, samples_count: int) -> None:
        if samples_count <= 0:
            return
        if self._realtime_start is None:
            self._realtime_start = time.monotonic()
            self._realtime_samples_sent = 0
        self._realtime_samples_sent += samples_count
        target_elapsed = self._realtime_samples_sent / sample_rate
        elapsed = time.monotonic() - self._realtime_start
        sleep_time = target_elapsed - elapsed - self._ahead_buffer_seconds
        if sleep_time > 0:
            time.sleep(sleep_time)

    def _open_track(self, track: TrackInfo) -> tuple[int | None, Iterator[np.ndarray]]:
        logging.info(f"Opening track: {track.path}")
        try:
            container = av.open(track.path)
        except Exception as e:
            logging.error(f"Failed to open track {track.path}: {e}")
            return None, iter(())

        stream = container.streams.audio[0]
        total_samples: int | None = None
        if stream.duration is not None and stream.time_base is not None:
            duration_seconds = float(stream.duration * stream.time_base)
            total_samples = int(duration_seconds * sample_rate)

        def iterator() -> Iterator[np.ndarray]:
            try:
                for frame in container.decode(stream):
                    frames = self._resampler.resample(frame)
                    for res_frame in frames:
                        yield res_frame.to_ndarray()
            except Exception as e:
                logging.error(f"Decode error on {track.path}: {e}")
            finally:
                container.close()

        return total_samples, iterator()

    def _prefetch_head(
        self, frames: Iterator[np.ndarray]
    ) -> tuple[np.ndarray, Iterator[np.ndarray]]:
        needed = self._buffer_samples
        chunks: list[np.ndarray] = []
        collected = 0

        try:
            for chunk in frames:
                chunks.append(chunk)
                collected += chunk.shape[1]
                if collected >= needed:
                    break
        except StopIteration:
            pass

        if not chunks:
            return np.zeros((channels, 0), dtype=np.float32), iter(())

        head = np.concatenate(chunks, axis=1)

        if head.shape[1] > needed:
            remainder_chunk = head[:, needed:]
            head = head[:, :needed]
            return head, itertools.chain([remainder_chunk], frames)

        return head, frames

    @dataclass
    class _QueuedTrack:
        track: TrackInfo
        total_samples: int | None
        head: np.ndarray
        frames: Iterator[np.ndarray]

    def _queue_next_track(
        self, next_track_provider: Callable[[], TrackInfo | None]
    ) -> _QueuedTrack | None:
        next_track = next_track_provider()
        if not next_track:
            return None
        logging.info(f"Next song queued: {next_track.title}")
        next_total, raw_iter = self._open_track(next_track)
        next_head, next_frames_iter = self._prefetch_head(raw_iter)
        return self._QueuedTrack(
            track=next_track,
            total_samples=next_total,
            head=next_head,
            frames=next_frames_iter,
        )

    def _crossfade_to_next(
        self, current_track: TrackInfo, queued: _QueuedTrack
    ) -> bool:
        buffered = self._peek_queue()

        # Trim trailing silence before making any decisions.
        buffered = self._trim_silence(buffered)

        head = queued.head
        length = min(buffered.shape[1], head.shape[1])
        if length == 0:
            return False

        fade_out_smart = self._detect_fade(buffered, direction="out")
        fade_in_smart = self._detect_fade(head, direction="in")

        if fade_out_smart and fade_in_smart:
            logging.info(
                f"Crossfading (Buffer length: {buffered.shape[1]/sample_rate:.2f}s)"
            )

            curr_gain = math.pow(10.0, current_track.gain_db / 20.0)
            next_gain = math.pow(10.0, queued.track.gain_db / 20.0)

            mixed = self._mix_crossfade(buffered * curr_gain, head * next_gain)

            self._mp3_conn.set_title(queued.track.title)
            self._encode_samples_scaled(mixed)

            self._clear_queue()
            if length < head.shape[1]:
                head_remainder = head[:, length:]
                if head_remainder.shape[1] > 0:
                    queued.frames = itertools.chain([head_remainder], queued.frames)
            return True

        return False

    def _drain_queue_without_crossfade(self, *, gain_db: float) -> None:
        logging.info("No crossfade. Draining queue (with silence trim).")
        buffered = self._peek_queue()
        buffered = self._trim_silence(buffered)
        self._encode_samples(buffered, gain_db=gain_db)
        self._clear_queue()

    def _start_next_track(
        self, queued: _QueuedTrack
    ) -> tuple[TrackInfo, int | None, Iterator[np.ndarray]]:
        self._mp3_conn.set_title(queued.track.title)
        for ready in self._enqueue_samples(queued.head):
            self._encode_samples(ready, gain_db=queued.track.gain_db)
        return queued.track, queued.total_samples, queued.frames

    def stream_tracks(
        self, next_track_provider: Callable[[], TrackInfo | None]
    ) -> None:
        current_track = next_track_provider()
        if not current_track:
            return

        current_total, current_frames = self._open_track(current_track)

        while current_track:
            logging.info(f"Now playing: {current_track.title}")
            self._mp3_conn.set_title(current_track.title)

            duration_known = current_total is not None
            total_samples = current_total if current_total else 0
            samples_sent = 0

            queued_next: AudioPipeline._QueuedTrack | None = None

            # --- Main Processing Loop for Current Track ---
            for chunk in current_frames:
                samples_sent += chunk.shape[1]
                for ready in self._enqueue_samples(chunk):
                    self._encode_samples(ready, gain_db=current_track.gain_db)

                if (
                    duration_known
                    and queued_next is None
                    and (total_samples - samples_sent) <= (self._buffer_samples * 1.5)
                ):
                    queued_next = self._queue_next_track(next_track_provider)

            # --- Crossfade Decision ---
            if duration_known and queued_next:
                if self._crossfade_to_next(current_track, queued_next):
                    current_track = queued_next.track
                    current_total = queued_next.total_samples
                    current_frames = queued_next.frames
                    continue

            # --- No Crossfade / Fallback ---
            self._drain_queue_without_crossfade(gain_db=current_track.gain_db)

            if queued_next:
                current_track, current_total, current_frames = self._start_next_track(
                    queued_next
                )
            else:
                current_track = next_track_provider()
                if current_track:
                    current_total, current_frames = self._open_track(current_track)
