from __future__ import annotations

import argparse
import asyncio
import logging
import math
import queue
import sys
import time
from dataclasses import dataclass
from threading import Event, Thread
from typing import Callable, Iterator, Sequence, cast
import itertools

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
        self.mount_name = mount.mount
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
        
        logging.info(f"Connecting to Icecast mount {mount.mount}...")
        conn.open()
        self._conn = conn
        self._allow_metadata = allow_metadata

    def send(self, data: bytes) -> None:
        if data:
            try:
                self._conn.send(data)
                self._conn.sync() 
            except Exception as e:
                logging.error(f"Error sending to {self.mount_name}: {e}")

    def set_title(self, title: str | None) -> None:
        if not self._allow_metadata or not title:
            return
        try:
            metadata = shout.Metadata()
            metadata.set("title", title)
            self._conn.metadata = metadata
        except Exception as e:
            logging.warning(f"Failed to set metadata: {e}")

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass


class ThreadedSender:
    """
    Decouples audio encoding from network transmission.
    Writes to a queue; background thread reads queue and sends to Icecast.
    """
    def __init__(self, conn: IcecastConnection, buffer_size: int = 500) -> None:
        self._conn = conn
        self._queue: queue.Queue[bytes | None] = queue.Queue(maxsize=buffer_size)
        self._stop_event = Event()
        self._thread = Thread(target=self._worker, daemon=True, name=f"Sender-{conn.mount_name}")
        self._thread.start()

    def _worker(self) -> None:
        while not self._stop_event.is_set():
            try:
                data = self._queue.get(timeout=1.0)
                if data is None:
                    break
                self._conn.send(data)
                self._queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Sender thread error: {e}")

    def write(self, data: bytes) -> int:
        try:
            self._queue.put(data, timeout=5.0) # Backpressure if net is dead
            return len(data)
        except queue.Full:
            logging.warning("Network buffer full! Dropping audio packet.")
            return len(data)

    def flush(self) -> None:
        pass

    def close(self) -> None:
        self._stop_event.set()
        try:
            self._queue.put(None, timeout=1.0)
        except queue.Full:
            pass
        self._thread.join(timeout=2.0)
        self._conn.close()

    # File-like interface for PyAV
    def readable(self) -> bool: return False
    def writable(self) -> bool: return True
    def seekable(self) -> bool: return False


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
        self._sender = ThreadedSender(conn)
        # PyAV writes to our ThreadedSender which looks like a file
        self._container = av.open(self._sender, mode="w", format=fmt)
        
        stream = cast(
            AudioStream,
            self._container.add_stream(codec_name, rate=sample_rate),
        )
        stream.channels = channels
        stream.layout = "stereo" if channels == 2 else "mono"
        stream.bit_rate = bitrate
        self._stream = stream

    def encode(self, frame: av.AudioFrame | None) -> None:
        # Note: frame=None triggers flush in PyAV
        for packet in self._stream.encode(frame):
            self._container.mux(packet)

    def close(self) -> None:
        self.encode(None)  # Flush encoder
        self._container.close()
        self._sender.close()


class AudioPipeline:
    def __init__(self, config: StreamConfig) -> None:
        self._config = config
        self._tail_samples = int(
            config.sample_rate * max(config.crossfade_seconds, config.buffer_seconds)
        )
        self._tail: np.ndarray | None = None
        self._layout = "stereo" if config.channels == 2 else "mono"
        
        self._resampler = AudioResampler(
            format="fltp",
            layout=self._layout,
            rate=config.sample_rate,
        )

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
        self._mp3_conn = mp3_conn

    def close(self) -> None:
        for encoder in self._encoders:
            encoder.close()

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
    
    def _trim_silence(self, samples: np.ndarray, threshold_db: float = -60.0) -> np.ndarray:
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
            logging.debug(f"Trimmed {trimmed} silent samples ({trimmed/self._config.sample_rate:.2f}s) from tail.")
            return samples[:, :last_audible]
            
        return samples

    def _detect_fade(self, samples: np.ndarray, *, direction: str) -> bool:
        if samples.shape[1] < (self._config.sample_rate * 0.5): # Need at least 0.5s to judge fade
            return False
        
        # Analyze RMS energy
        window = max(1, int(samples.shape[1] / 20))
        rms: list[float] = []
        for i in range(0, samples.shape[1], window):
            chunk = samples[:, i : i + window]
            if chunk.shape[1] == 0: continue
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
            return np.zeros((self._config.channels, 0), dtype=np.float32)
            
        fade_out = np.linspace(1.0, 0.0, num=length, dtype=np.float32)
        fade_in = np.linspace(0.0, 1.0, num=length, dtype=np.float32)
        
        mixed = tail[:, -length:] * fade_out + head[:, :length] * fade_in
        return mixed

    def _encode_samples_scaled(self, samples: np.ndarray) -> None:
        if samples.shape[1] == 0: return
        samples = np.clip(samples, -1.0, 1.0)
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
            total_samples = int(duration_seconds * self._config.sample_rate)

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
        needed = self._tail_samples
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
            return np.zeros((self._config.channels, 0), dtype=np.float32), iter(())
            
        head = np.concatenate(chunks, axis=1)
        
        if head.shape[1] > needed:
            remainder_chunk = head[:, needed:]
            head = head[:, :needed]
            return head, itertools.chain([remainder_chunk], frames)
            
        return head, frames

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

            next_track: TrackInfo | None = None
            next_head: np.ndarray | None = None
            next_frames_iter: Iterator[np.ndarray] | None = None
            next_total: int | None = None

            # --- Main Processing Loop for Current Track ---
            for chunk in current_frames:
                samples_sent += chunk.shape[1]
                for ready in self._push_samples(chunk):
                    self._encode_samples(ready, gain_db=current_track.gain_db)

                if (
                    duration_known
                    and next_track is None
                    and (total_samples - samples_sent) <= (self._tail_samples * 1.5)
                ):
                    next_track = next_track_provider()
                    if next_track:
                        logging.info(f"Next song queued: {next_track.title}")
                        next_total, raw_iter = self._open_track(next_track)
                        next_head, next_frames_iter = self._prefetch_head(raw_iter)

            # --- Crossfade Decision ---
            crossfade_occured = False
            
            if duration_known and next_track and next_head is not None and next_frames_iter is not None:
                tail = self._tail if self._tail is not None else np.zeros((self._config.channels, 0), dtype=np.float32)
                
                # 1. TRIM SILENCE: This is key. We remove trailing zeros.
                # If the song had 4s of silence at the end, 'tail' is now 4s shorter.
                tail = self._trim_silence(tail)
                
                # 2. DECIDE: Smart detect or just always crossfade?
                # Even if detection fails, a small crossfade is usually better than a hard cut.
                fade_out_smart = self._detect_fade(tail, direction="out")
                fade_in_smart = self._detect_fade(next_head, direction="in")
                
                # If we have any audio left in tail after trimming, we crossfade.
                if tail.shape[1] > 0:
                    logging.info(f"Crossfading (Tail length: {tail.shape[1]/self._config.sample_rate:.2f}s)")
                    
                    curr_gain = math.pow(10.0, (self._config.gain_db + current_track.gain_db) / 20.0)
                    next_gain = math.pow(10.0, (self._config.gain_db + next_track.gain_db) / 20.0)
                    
                    # Mix using the potentially shorter tail
                    mixed = self._mix_crossfade(tail * curr_gain, next_head * next_gain)
                    
                    self._mp3_conn.set_title(next_track.title)
                    self._encode_samples_scaled(mixed)
                    
                    self._tail = None
                    current_track = next_track
                    current_total = next_total
                    current_frames = next_frames_iter
                    crossfade_occured = True
            
            if crossfade_occured:
                continue

            # --- No Crossfade / Fallback ---
            # Even here, we trim silence. If we are just playing songs back-to-back,
            # we don't want to broadcast 4 seconds of silence.
            logging.info("No crossfade. Draining tail (with silence trim).")
            
            # 1. Get the tail
            tail = self._tail if self._tail is not None else np.zeros((self._config.channels, 0), dtype=np.float32)
            
            # 2. Trim silence from it
            tail = self._trim_silence(tail)
            
            # 3. Play the trimmed tail
            self._encode_samples(tail, gain_db=current_track.gain_db)
            self._tail = None # Clear buffer

            if next_track and next_frames_iter:
                # Play the pre-fetched head
                self._mp3_conn.set_title(next_track.title)
                for ready in self._push_samples(next_head): # type: ignore
                     self._encode_samples(ready, gain_db=next_track.gain_db)
                
                current_track = next_track
                current_total = next_total
                current_frames = next_frames_iter
            else:
                current_track = next_track_provider()
                if current_track:
                    current_total, current_frames = self._open_track(current_track)
async def stream_forever(config: StreamConfig) -> None:
    pipeline = AudioPipeline(config)
    loop = asyncio.get_running_loop()
    stop_event = Event()

    def next_track_blocking() -> TrackInfo | None:
        if stop_event.is_set():
            return None
        future = asyncio.run_coroutine_threadsafe(get_next_track(), loop)
        try:
            return future.result()
        except Exception as e:
            logging.error(f"Error fetching next track: {e}")
            return None

    worker = Thread(
        target=pipeline.stream_tracks, args=(next_track_blocking,), daemon=True
    )
    worker.start()
    try:
        while worker.is_alive():
            await asyncio.sleep(0.5)
    except asyncio.CancelledError:
        pass
    finally:
        logging.info("Shutting down streamer...")
        stop_event.set()
        pipeline.close() # This will join the sender threads
        worker.join(timeout=2.0)


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
            name="My MP3 Stream",
            description="Python Generated",
            genre="Various",
            url=None,
            public=0,
        ),
        opus=StreamMount(
            mount=args.opus_mount,
            bitrate=args.opus_bitrate,
            name="My Opus Stream",
            description="Python Generated",
            genre="Various",
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
    except KeyboardInterrupt:
        pass
    except Exception:
        logging.exception("Streamer crashed.")
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
