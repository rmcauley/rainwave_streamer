import asyncio
import logging
import sys
import time
import tracemalloc
from concurrent.futures import Future
from threading import Event, Lock, Thread
from typing import Literal

import psutil

import streamer.audio_pipeline as audio_pipeline_module
from streamer.audio_pipeline import AudioPipeline, AudioPipelineGracefulShutdownError
from streamer.decoders.gstreamer_audio_track import AudioTrackInfo
from streamer.get_next_track_from_rainwave import (
    get_next_track_from_rainwave,
    mark_track_invalid_on_rainwave,
)
from streamer.stream_config import StreamConfig
from streamer.stream_constants import mp3_bitrate_approx, opus_bitrate_approx
from streamer.stream_mount import StreamMount

memory_log_interval_seconds = 30.0
bytes_per_mebibyte = 1024 * 1024
tracemalloc_frames = 25


class NullSinkConnection:
    """
    Drop-in replacement for IcecastConnection that discards encoded bytes.
    """

    def __init__(
        self,
        config: StreamConfig,
        mount: StreamMount,
        *,
        fmt: Literal["mp3", "ogg"],
    ) -> None:
        del config
        del fmt
        self.mount_name = mount.mount
        self._closed = False
        self._bytes_sent = 0
        self._lock = Lock()

    def send(self, data: bytes) -> None:
        if not data:
            return
        with self._lock:
            if self._closed:
                return
            self._bytes_sent += len(data)

    def close(self) -> None:
        with self._lock:
            self._closed = True


def _build_null_stream_config() -> StreamConfig:
    return StreamConfig(
        host="127.0.0.1",
        port=8000,
        password="unused",
        mp3=StreamMount(
            mount="null_sink.mp3",
            bitrate=mp3_bitrate_approx,
            name="Null Sink Memory Leak Test",
            description="Local sink test; no Icecast connection.",
            genre="Test",
            url="http://localhost/null_sink.mp3",
            public=0,
        ),
        opus=StreamMount(
            mount="null_sink.ogg",
            bitrate=opus_bitrate_approx,
            name="Null Sink Memory Leak Test",
            description="Local sink test; no Icecast connection.",
            genre="Test",
            url="http://localhost/null_sink.ogg",
            public=0,
        ),
    )


async def stream_forever_to_null_sink() -> None:
    loop = asyncio.get_running_loop()
    shutdown_requested = Event()
    worker_error: list[BaseException] = []
    pipeline: AudioPipeline | None = None
    worker: Thread | None = None

    if not tracemalloc.is_tracing():
        tracemalloc.start(tracemalloc_frames)
    process = psutil.Process()
    next_memory_log_at = 0.0

    def log_memory_usage(force: bool = False) -> None:
        nonlocal next_memory_log_at

        now = time.monotonic()
        if not force and now < next_memory_log_at:
            return
        next_memory_log_at = now + memory_log_interval_seconds

        try:
            rss_bytes = process.memory_info().rss
            thread_count = process.num_threads()
            traced_current, traced_peak = tracemalloc.get_traced_memory()
        except Exception as e:
            logging.debug("Unable to collect memory usage stats: %s", e)
            return

        logging.info(
            "Memory usage: rss=%.1fMiB python_current=%.1fMiB python_peak=%.1fMiB threads=%d",
            rss_bytes / bytes_per_mebibyte,
            traced_current / bytes_per_mebibyte,
            traced_peak / bytes_per_mebibyte,
            thread_count,
        )

    def should_stop_workers() -> bool:
        return shutdown_requested.is_set()

    def next_track_blocking() -> AudioTrackInfo:
        if should_stop_workers():
            raise AudioPipelineGracefulShutdownError()
        future = asyncio.run_coroutine_threadsafe(get_next_track_from_rainwave(), loop)
        try:
            track_info = future.result(timeout=2.0)
            if track_info is None:
                raise RuntimeError("No track info returned by get_next_track.")
            return track_info
        except Exception as e:
            future.cancel()
            if should_stop_workers():
                raise AudioPipelineGracefulShutdownError() from e
            raise RuntimeError("Error fetching next track") from e

    def mark_track_invalid_fire_and_forget(path: str) -> None:
        if should_stop_workers():
            return
        try:
            future = asyncio.run_coroutine_threadsafe(
                mark_track_invalid_on_rainwave(path), loop
            )
        except Exception:
            return

        def _ignore_mark_result(done_future: Future[None]) -> None:
            try:
                done_future.result()
            except Exception:
                pass

        future.add_done_callback(_ignore_mark_result)

    def worker_target() -> None:
        if pipeline is None:
            return
        try:
            pipeline.stream_tracks()
        except AudioPipelineGracefulShutdownError:
            shutdown_requested.set()
        except Exception as e:
            worker_error.append(e)
            shutdown_requested.set()

    original_connection_class = audio_pipeline_module.IcecastConnection
    audio_pipeline_module.IcecastConnection = NullSinkConnection  # type: ignore[assignment]

    try:
        pipeline = AudioPipeline(
            _build_null_stream_config(),
            next_track_blocking,
            mark_track_invalid_fire_and_forget,
            should_stop_workers,
        )
        worker = Thread(target=worker_target, daemon=True, name="AudioPipelineWorker")
        worker.start()

        while worker.is_alive():
            log_memory_usage()
            if worker_error:
                raise worker_error[0]
            await asyncio.sleep(0.2)

        if worker_error:
            raise worker_error[0]
    except asyncio.CancelledError:
        shutdown_requested.set()
        raise
    finally:
        log_memory_usage(force=True)
        shutdown_requested.set()
        audio_pipeline_module.IcecastConnection = original_connection_class

        worker_stopped = worker is None
        if worker is not None:
            worker.join(timeout=2.0)
            worker_stopped = not worker.is_alive()
        if not worker_stopped:
            logging.warning(
                "Worker did not stop within 2 seconds; continuing shutdown."
            )


def main() -> int:
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    try:
        asyncio.run(stream_forever_to_null_sink())
    except KeyboardInterrupt:
        pass
    except Exception:
        logging.exception("Null sink memory test crashed.")
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
