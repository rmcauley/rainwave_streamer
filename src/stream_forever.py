#########################################################################
# This file was AI generated with no human supervision.
#########################################################################

import asyncio
import logging
import signal
from concurrent.futures import Future
from threading import Event, Thread

from streamer.get_next_track_from_rainwave import (
    get_next_track_from_rainwave,
    mark_track_invalid_on_rainwave,
)
from streamer.audio_pipeline import AudioPipeline, AudioPipelineGracefulShutdownError
from streamer.audio_track import AudioTrackInfo
from streamer.stream_config import StreamConfig


async def stream_forever(config: StreamConfig) -> None:
    loop = asyncio.get_running_loop()
    shutdown_requested = Event()
    shutting_down = Event()
    worker_error: list[BaseException] = []
    pipeline: AudioPipeline | None = None
    worker: Thread | None = None
    installed_signal_handlers: list[signal.Signals] = []

    def should_stop_workers() -> bool:
        return shutdown_requested.is_set()

    def next_track_blocking() -> AudioTrackInfo:
        if should_stop_workers():
            raise AudioPipelineGracefulShutdownError()
        future = asyncio.run_coroutine_threadsafe(get_next_track_from_rainwave(), loop)
        try:
            # Review note: this strict timeout is intentional to fail fast.
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
            if shutting_down.is_set():
                logging.warning(
                    "Worker exception during shutdown (suppressed): %s",
                    e,
                    exc_info=e,
                )
                return
            worker_error.append(e)
            shutdown_requested.set()

    def handle_intentional_signal(signal_name: str) -> None:
        logging.info(
            "Received %s. Starting intentional shutdown handling.", signal_name
        )
        shutdown_requested.set()

    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, handle_intentional_signal, sig.name)
                installed_signal_handlers.append(sig)
            except (NotImplementedError, RuntimeError, ValueError):
                pass

        pipeline = AudioPipeline(
            config,
            next_track_blocking,
            mark_track_invalid_fire_and_forget,
            should_stop_workers,
        )
        worker = Thread(target=worker_target, daemon=True, name="AudioPipelineWorker")
        worker.start()
        while worker.is_alive():
            if worker_error:
                raise worker_error[0]
            await asyncio.sleep(0.2)
        if worker_error:
            raise worker_error[0]
    except asyncio.CancelledError:
        shutdown_requested.set()
        raise
    finally:
        shutting_down.set()
        shutdown_requested.set()

        # Review note: this daemon intentionally allows repeated SIGINT/SIGTERM to
        # short-circuit graceful shutdown if the operator insists.
        for installed_sig in installed_signal_handlers:
            try:
                loop.remove_signal_handler(installed_sig)
            except Exception:
                pass

        worker_stopped = worker is None
        if worker is not None:
            worker.join(timeout=2.0)
            worker_stopped = not worker.is_alive()
        if not worker_stopped:
            logging.warning(
                "Worker did not stop within 2 seconds; continuing shutdown."
            )
