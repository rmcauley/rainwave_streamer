from __future__ import annotations

import argparse
import asyncio
import logging
import math
import sys
from dataclasses import dataclass
from threading import Event, Thread
from typing import Callable, Iterator, Sequence, cast

import av
from av.audio.resampler import AudioResampler
from av.audio.stream import AudioStream
import numpy as np
import shout


@dataclass(frozen=True)
class TrackInfo:
    path: str
    title: str | None = None
    gain_db: float = 0.0


@dataclass(frozen=True)
class StreamMount:
    mount: str
    bitrate: int
    name: str
    description: str | None
    genre: str | None
    url: str | None
    public: int


@dataclass(frozen=True)
class StreamConfig:
    host: str
    port: int
    user: str
    password: str
    mp3: StreamMount
    opus: StreamMount
    sample_rate: int
    channels: int
    crossfade_seconds: float
    buffer_seconds: float
    gain_db: float


async def get_next_track() -> TrackInfo:
    """
    Placeholder for caller-supplied async function.
    Replace with your own logic (DB, queue, etc.).
    Provide title (for MP3 metadata only) and gain_db in +/- dB.
    """
    raise NotImplementedError("Implement get_next_track() in your application.")


class IcecastConnection:
    def __init__(
        self,
        config: StreamConfig,
        mount: StreamMount,
        *,
        fmt: int,
        allow_metadata: bool,
    ) -> None:
        conn = shout.Shout()
        conn.host = config.host
        conn.port = config.port
        conn.user = config.user
        conn.password = config.password
        conn.mount = mount.mount
        conn.format = fmt
        conn.protocol = shout.PROTOCOL_HTTP
        conn.public = mount.public
        conn.name = mount.name
        if mount.description:
            conn.description = mount.description
        if mount.genre:
            conn.genre = mount.genre
        if mount.url:
            conn.url = mount.url
        conn.open()
        self._conn = conn
        self._allow_metadata = allow_metadata

    def send(self, data: bytes) -> None:
        if data:
            self._conn.send(data)

    def set_title(self, title: str | None) -> None:
        if not self._allow_metadata or not title:
            return
        metadata = shout.Metadata()
        metadata.set("title", title)
        self._conn.metadata = metadata

    def close(self) -> None:
        self._conn.close()


class ShoutIO:
    def __init__(self, conn: IcecastConnection) -> None:
        self._conn = conn

    def write(self, data: bytes) -> int:
        self._conn.send(data)
        return len(data)

    def flush(self) -> None:  # pragma: no cover - protocol hook
        return None

    def close(self) -> None:  # pragma: no cover - protocol hook
        return None

    def readable(self) -> bool:  # pragma: no cover - protocol hook
        return False

    def writable(self) -> bool:  # pragma: no cover - protocol hook
        return True

    def seekable(self) -> bool:  # pragma: no cover - protocol hook
        return False


class Encoder:
    def __init__(
        self,
        conn: IcecastConnection,
        *,
        codec_name: str,
        fmt: str,
        sample_rate: int,
        channels: int,
        bitrate: int,
    ) -> None:
        # PyAV writes to file-like objects; we adapt libshout's send() to a file-like API.
        self._io = ShoutIO(conn)
        self._container = av.open(self._io, mode="w", format=fmt)
        stream = cast(
            AudioStream,
            self._container.add_stream(  # pyright: ignore[reportUnknownMemberType]
                codec_name, rate=sample_rate
            ),
        )
        stream.channels = channels
        stream.layout = "stereo" if channels == 2 else "mono"
        stream.bit_rate = bitrate
        self._stream = stream

    def encode(self, frame: av.AudioFrame) -> None:
        for packet in self._stream.encode(frame):
            self._container.mux(packet)

    def flush(self) -> None:
        for packet in self._stream.encode(None):
            self._container.mux(packet)

    def close(self) -> None:
        self.flush()
        self._container.close()


class AudioPipeline:
    def __init__(self, config: StreamConfig) -> None:
        self._config = config
        self._tail_samples = int(
            config.sample_rate * max(config.crossfade_seconds, config.buffer_seconds)
        )
        self._tail: np.ndarray | None = None
        self._layout = "stereo" if config.channels == 2 else "mono"
        # PyAV resampler turns decoded audio into float planar numpy arrays.
        self._resampler = AudioResampler(
            format="fltp",
            layout=self._layout,
            rate=config.sample_rate,
        )

        # libshout manages the persistent Icecast connections.
        mp3_conn = IcecastConnection(
            config, config.mp3, fmt=shout.FORMAT_MP3, allow_metadata=True
        )
        opus_conn = IcecastConnection(
            config, config.opus, fmt=shout.FORMAT_OGG, allow_metadata=False
        )
        self._encoders = (
            Encoder(
                mp3_conn,
                codec_name="libmp3lame",
                fmt="mp3",
                sample_rate=config.sample_rate,
                channels=config.channels,
                bitrate=config.mp3.bitrate,
            ),
            Encoder(
                opus_conn,
                codec_name="libopus",
                fmt="ogg",
                sample_rate=config.sample_rate,
                channels=config.channels,
                bitrate=config.opus.bitrate,
            ),
        )
        self._connections = (mp3_conn, opus_conn)
        self._mp3_conn = mp3_conn

    def close(self) -> None:
        for encoder in self._encoders:
            encoder.close()
        for conn in self._connections:
            conn.close()

    def _push_samples(self, samples: np.ndarray) -> Iterator[np.ndarray]:
        if self._tail is None:
            self._tail = samples
            return iter(())
        combined = np.concatenate([self._tail, samples], axis=1)
        if combined.shape[1] <= self._tail_samples:
            self._tail = combined
            return iter(())
        split = combined.shape[1] - self._tail_samples
        ready = combined[:, :split]
        self._tail = combined[:, split:]
        return iter((ready,))

    def _drain_tail(self) -> Iterator[np.ndarray]:
        if self._tail is None:
            return iter(())
        ready = self._tail
        self._tail = None
        return iter((ready,))

    def _detect_fade(self, samples: np.ndarray, *, direction: str) -> bool:
        if samples.shape[1] < self._tail_samples:
            return False
        window = max(1, int(self._tail_samples / 10))
        rms: list[float] = []
        for i in range(0, self._tail_samples, window):
            chunk = samples[:, i : i + window]
            rms.append(float(np.sqrt(np.mean(chunk * chunk))))
        if len(rms) < 2:
            return False
        slope = np.polyfit(np.arange(len(rms)), rms, 1)[0]
        threshold = max(1e-6, 0.015 * max(rms))
        if direction == "out":
            return slope < -threshold
        return slope > threshold

    def _mix_crossfade(self, tail: np.ndarray, head: np.ndarray) -> np.ndarray:
        length = min(tail.shape[1], head.shape[1])
        fade_out = np.linspace(1.0, 0.0, num=length, dtype=np.float32)
        fade_in = np.linspace(0.0, 1.0, num=length, dtype=np.float32)
        mixed = tail[:, -length:] * fade_out + head[:, :length] * fade_in
        return mixed

    def _start_next_track_fetch(
        self, provider: Callable[[], TrackInfo | None]
    ) -> tuple[Thread, dict[str, TrackInfo | None]]:
        result: dict[str, TrackInfo | None] = {"track": None}

        def runner() -> None:
            result["track"] = provider()

        thread = Thread(target=runner, daemon=True)
        thread.start()
        return thread, result

    def _encode_samples_scaled(self, samples: np.ndarray) -> None:
        samples = np.clip(samples, -1.0, 1.0)
        # numpy arrays represent audio as shape (channels, samples).
        frame = av.AudioFrame.from_ndarray(
            samples.astype(np.float32, copy=False),
            format="fltp",
            layout=self._layout,
        )
        frame.sample_rate = self._config.sample_rate
        for encoder in self._encoders:
            encoder.encode(frame)

    def _encode_samples(self, samples: np.ndarray, *, gain_db: float) -> None:
        gain = math.pow(10.0, (self._config.gain_db + gain_db) / 20.0)
        self._encode_samples_scaled(samples * gain)

    def _open_track(self, track: TrackInfo) -> tuple[int | None, Iterator[np.ndarray]]:
        container = av.open(track.path)
        stream = container.streams.audio[0]
        total_samples: int | None = None
        if stream.duration is not None and stream.time_base is not None:
            duration_seconds = float(stream.duration * stream.time_base)
            total_samples = int(duration_seconds * self._config.sample_rate)

        def iterator() -> Iterator[np.ndarray]:
            try:
                for frame in container.decode(stream):
                    frames = self._resampler.resample(frame)
                    for res_frame in frames:
                        # Each resampled frame becomes a float32 numpy array.
                        array = res_frame.to_ndarray()
                        yield array
            finally:
                container.close()

        return total_samples, iterator()

    def _prefetch_head(
        self, frames: Iterator[np.ndarray]
    ) -> tuple[np.ndarray, Iterator[np.ndarray]]:
        needed = self._tail_samples
        chunks: list[np.ndarray] = []
        collected = 0
        for chunk in frames:
            chunks.append(chunk)
            collected += chunk.shape[1]
            if collected >= needed:
                break
        if not chunks:
            return np.zeros((self._config.channels, 0), dtype=np.float32), iter(())
        head = np.concatenate(chunks, axis=1)
        if head.shape[1] > needed:
            remainder = head[:, needed:]
            head = head[:, :needed]

            def remainder_iter() -> Iterator[np.ndarray]:
                yield remainder
                yield from frames

            return head, remainder_iter()
        return head, frames

    def stream_tracks(
        self, next_track_provider: Callable[[], TrackInfo | None]
    ) -> None:
        current = next_track_provider()
        if current is None:
            return

        while current:
            logging.info("Now playing: %s", current.path)
            logging.info(
                "Track gain (dB): %.2f (base %.2f)",
                current.gain_db,
                self._config.gain_db,
            )
            self._mp3_conn.set_title(current.title)
            potential_total_samples, frame_iter = self._open_track(current)
            duration_known = potential_total_samples is not None
            total_samples = (
                potential_total_samples if potential_total_samples is not None else 0
            )
            samples_sent = 0

            next_track: TrackInfo | None = None
            next_head: np.ndarray | None = None
            next_frames: Iterator[np.ndarray] | None = None

            for chunk in frame_iter:
                samples_sent += chunk.shape[1]
                for ready in self._push_samples(chunk):
                    self._encode_samples(ready, gain_db=current.gain_db)

                if (
                    duration_known
                    and next_track is None
                    and total_samples - samples_sent <= self._tail_samples
                ):
                    next_track = next_track_provider()
                    if next_track:
                        logging.info("Next song queued: %s", next_track.path)
                        _total, next_frames_iter = self._open_track(next_track)
                        next_head, next_frames = self._prefetch_head(next_frames_iter)

            if duration_known and next_track and next_head is not None:
                tail = (
                    self._tail
                    if self._tail is not None
                    else np.zeros((self._config.channels, 0), dtype=np.float32)
                )
                if tail.shape[1] and next_head.shape[1]:
                    fade_out = self._detect_fade(tail, direction="out")
                    fade_in = self._detect_fade(next_head, direction="in")
                    if fade_out and fade_in:
                        logging.info(
                            "Crossfade: enabled (fade-out + fade-in detected)."
                        )
                        current_gain = math.pow(
                            10.0, (self._config.gain_db + current.gain_db) / 20.0
                        )
                        next_gain = math.pow(
                            10.0, (self._config.gain_db + next_track.gain_db) / 20.0
                        )
                        mixed = self._mix_crossfade(
                            tail * current_gain,
                            next_head * next_gain,
                        )
                        self._mp3_conn.set_title(next_track.title)
                        self._encode_samples_scaled(mixed)
                        self._tail = None
                        if next_frames is not None:
                            for chunk in next_frames:
                                for ready in self._push_samples(chunk):
                                    self._encode_samples(
                                        ready, gain_db=next_track.gain_db
                                    )
                        current = next_track
                        continue
                    logging.info("Crossfade: disabled (fade pattern not detected).")

            if not duration_known:
                logging.info("Unknown duration, disabling crossfade for this track.")
                fetch_thread, fetch_result = self._start_next_track_fetch(
                    next_track_provider
                )
                for ready in self._drain_tail():
                    self._encode_samples(ready, gain_db=current.gain_db)
                fetch_thread.join()
                current = fetch_result["track"]
                continue

            for ready in self._drain_tail():
                self._encode_samples(ready, gain_db=current.gain_db)

            if next_track is not None and next_frames is not None:
                self._mp3_conn.set_title(next_track.title)
                for chunk in next_frames:
                    for ready in self._push_samples(chunk):
                        self._encode_samples(ready, gain_db=next_track.gain_db)
                current = next_track
            else:
                current = next_track_provider()


async def stream_forever(config: StreamConfig) -> None:
    pipeline = AudioPipeline(config)
    loop = asyncio.get_running_loop()
    stop_event = Event()

    def next_track_blocking() -> TrackInfo | None:
        if stop_event.is_set():
            return None
        future = asyncio.run_coroutine_threadsafe(get_next_track(), loop)
        return future.result()

    worker = Thread(
        target=pipeline.stream_tracks, args=(next_track_blocking,), daemon=True
    )
    worker.start()
    try:
        while worker.is_alive():
            await asyncio.sleep(0.5)
    finally:
        stop_event.set()
        worker.join()
        pipeline.close()


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Python-native Icecast streamer.")
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--user", default="source")
    parser.add_argument("--password", required=True)
    parser.add_argument("--mp3-mount", default="/stream.mp3")
    parser.add_argument("--opus-mount", default="/stream.opus")
    parser.add_argument("--mp3-bitrate", type=int, default=128000)
    parser.add_argument("--opus-bitrate", type=int, default=96000)
    parser.add_argument("--sample-rate", type=int, default=48000)
    parser.add_argument("--channels", type=int, default=2)
    parser.add_argument("--crossfade-seconds", type=float, default=5.0)
    parser.add_argument(
        "--buffer-seconds",
        type=float,
        default=2.0,
        help="Seconds of audio to keep buffered to cover track fetch latency.",
    )
    parser.add_argument("--gain-db", type=float, default=0.0)
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    config = StreamConfig(
        host=args.host,
        port=args.port,
        user=args.user,
        password=args.password,
        mp3=StreamMount(
            mount=args.mp3_mount,
            bitrate=args.mp3_bitrate,
            name="Rainwave MP3",
            description=None,
            genre=None,
            url=None,
            public=0,
        ),
        opus=StreamMount(
            mount=args.opus_mount,
            bitrate=args.opus_bitrate,
            name="Rainwave Opus",
            description=None,
            genre=None,
            url=None,
            public=0,
        ),
        sample_rate=args.sample_rate,
        channels=args.channels,
        crossfade_seconds=args.crossfade_seconds,
        buffer_seconds=args.buffer_seconds,
        gain_db=args.gain_db,
    )
    try:
        asyncio.run(stream_forever(config))
    except Exception:
        logging.exception("Streamer crashed.")
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
