#########################################################################
# This file was AI generated with no human supervision.
#########################################################################

import asyncio
import logging
from concurrent.futures import Future
from threading import Event, Thread

from streamer.get_next_track_from_rainwave import (
    get_next_track_from_rainwave,
    mark_track_invalid_on_rainwave,
)
from streamer.audio_pipeline import AudioPipeline
from streamer.audio_track import AudioTrackInfo
from streamer.stream_config import StreamConfig


async def stream_forever(config: StreamConfig) -> None:
    loop = asyncio.get_running_loop()
    stop_event = Event()
    worker_error: list[BaseException] = []
    pipeline: AudioPipeline | None = None
    worker: Thread | None = None

    def next_track_blocking() -> AudioTrackInfo:
        if stop_event.is_set():
            raise RuntimeError("Track fetch requested during shutdown.")
        future = asyncio.run_coroutine_threadsafe(get_next_track_from_rainwave(), loop)
        try:
            # Review note: this strict timeout is intentional to fail fast.
            track_info = future.result(timeout=2.0)
            if track_info is None:
                raise RuntimeError("No track info returned by get_next_track.")
            return track_info
        except Exception as e:
            future.cancel()
            raise RuntimeError("Error fetching next track") from e

    def mark_track_invalid_fire_and_forget(path: str) -> None:
        if stop_event.is_set():
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
            pipeline.stream_tracks(
                should_stop=stop_event.is_set,
            )
        except Exception as e:
            worker_error.append(e)
            stop_event.set()

    try:
        pipeline = AudioPipeline(
            config, next_track_blocking, mark_track_invalid_fire_and_forget
        )
        worker = Thread(target=worker_target, daemon=True)
        worker.start()
        while worker.is_alive():
            if worker_error:
                raise worker_error[0]
            await asyncio.sleep(0.5)
        if worker_error:
            raise worker_error[0]
    finally:
        logging.info("Shutting down streamer...")
        stop_event.set()
        worker_stopped = True
        if worker is not None:
            worker.join(timeout=2.0)
            if worker.is_alive():
                worker_stopped = False
                logging.warning("Pipeline worker did not stop within 2 seconds.")
        if pipeline is not None:
            if worker_stopped:
                pipeline.close()  # This will join the sender threads
            else:
                logging.warning(
                    "Skipping pipeline.close() because worker is still running."
                )
