import asyncio
import logging
import sys
import time
import tracemalloc
from concurrent.futures import Future
from threading import Event, Thread

import psutil

from streamer.audio_pipeline import AudioPipeline, AudioPipelineGracefulShutdownError
from streamer.connectors.null_sink_connection import NullSinkConnection
from streamer.decoders.audio_track import AudioTrackInfo
from streamer.decoders.gstreamer_audio_track import GstreamerAudioTrack
from streamer.encoder_senders.subprocess_encoder import SubprocessEncoderSender
from streamer.get_next_track_from_rainwave import (
    get_next_track_from_rainwave,
    mark_track_invalid_on_rainwave,
)
from streamer.stream_config import StreamConfig

# Swap these classes for performance or leak experiments.
DECODER_CLASS = GstreamerAudioTrack
ENCODER_SENDER_CLASS = SubprocessEncoderSender

memory_log_interval_seconds = 30.0
bytes_per_mebibyte = 1024 * 1024
tracemalloc_frames = 25


def _build_null_stream_config() -> StreamConfig:
    return StreamConfig(
        description="Local sink test; no Icecast connection.",
        genre="Test",
        host="127.0.0.1",
        name="Null Sink Memory Leak Test",
        password="unused",
        port=8000,
        stream_filename="null_sink",
        url="http://localhost/null_sink",
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

    try:
        pipeline = AudioPipeline(
            audio_track=DECODER_CLASS,
            config=_build_null_stream_config(),
            encoder_sender=ENCODER_SENDER_CLASS,
            get_next_track_from_rainwave=next_track_blocking,
            mark_track_invalid_on_rainwave=mark_track_invalid_fire_and_forget,
            server_connector=NullSinkConnection,
            should_stop=should_stop_workers,
            use_realtime_wait=False,
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
