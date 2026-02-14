import asyncio
import logging
import time
import tracemalloc
from concurrent.futures import Future
from threading import Event, Thread

import psutil

from streamer.audio_pipeline import AudioPipeline, AudioPipelineGracefulShutdownError
from streamer.sinks.sink import AudioSinkConstructor
from streamer.track_decoders.track_decoder import (
    AudioTrackDecoderConstructor,
    TrackInfo,
)
from streamer.encoder_senders.encoder_sender import EncoderSenderConstructor
from streamer.get_next_track_from_rainwave import (
    get_next_track_from_rainwave,
    mark_track_invalid_on_rainwave,
)
from streamer.stream_config import StreamConfig

memory_log_interval_seconds = 30.0
bytes_per_mebibyte = 1024 * 1024
tracemalloc_frames = 25


async def stream_forever(
    config: StreamConfig,
    decoder: AudioTrackDecoderConstructor,
    encoder: EncoderSenderConstructor,
    connection: AudioSinkConstructor,
    use_realtime_wait: bool,
    show_performance: bool,
) -> None:
    loop = asyncio.get_running_loop()
    shutdown_requested = Event()
    worker_error: list[BaseException] = []
    pipeline: AudioPipeline | None = None
    worker: Thread | None = None
    process: psutil.Process | None = None

    if show_performance and not tracemalloc.is_tracing():
        tracemalloc.start(tracemalloc_frames)
        process = psutil.Process()
        next_memory_log_at = 0.0

    def log_memory_usage(pipeline: AudioPipeline | None, force: bool = False) -> None:
        if process is None:
            return

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

        print(
            "Memory usage after %s songs: rss=%.1fMiB python_current=%.1fMiB python_peak=%.1fMiB threads=%d"
            % (
                pipeline.track_change_counter if pipeline is not None else "???",
                rss_bytes / bytes_per_mebibyte,
                traced_current / bytes_per_mebibyte,
                traced_peak / bytes_per_mebibyte,
                thread_count,
            )
        )

    def should_stop_workers() -> bool:
        return shutdown_requested.is_set()

    def next_track_blocking() -> TrackInfo:
        if should_stop_workers():
            raise AudioPipelineGracefulShutdownError()
        future = asyncio.run_coroutine_threadsafe(
            get_next_track_from_rainwave(config.sid), loop
        )
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
            audio_track=decoder,
            config=config,
            encoder_sender=encoder,
            get_next_track_from_rainwave=next_track_blocking,
            mark_track_invalid_on_rainwave=mark_track_invalid_fire_and_forget,
            server_connector=connection,
            should_stop=should_stop_workers,
            use_realtime_wait=use_realtime_wait,
            show_performance=show_performance,
        )
        worker = Thread(target=worker_target, daemon=True, name="AudioPipelineWorker")
        worker.start()

        while worker.is_alive():
            if show_performance:
                log_memory_usage(pipeline)
            if worker_error:
                raise worker_error[0]
            await asyncio.sleep(1)

        if worker_error:
            raise worker_error[0]
    except asyncio.CancelledError:
        shutdown_requested.set()
        raise
    finally:
        log_memory_usage(pipeline, force=True)
        shutdown_requested.set()

        worker_stopped = worker is None
        if worker is not None:
            worker.join(timeout=2.0)
            worker_stopped = not worker.is_alive()
        if not worker_stopped:
            logging.warning(
                "Worker did not stop within 2 seconds; continuing shutdown."
            )
