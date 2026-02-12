#########################################################################
# This file was AI generated with no human supervision.
#########################################################################

import asyncio
import logging
from threading import Event, Thread

from streamer.get_next_track import get_next_track
from streamer.audio_pipeline import AudioPipeline
from streamer.audio_track import AudioTrackInfo
from streamer.stream_config import StreamConfig


async def stream_forever(config: StreamConfig) -> None:
    pipeline = AudioPipeline(config)
    loop = asyncio.get_running_loop()
    stop_event = Event()
    worker_error: list[BaseException] = []

    def next_track_blocking() -> AudioTrackInfo:
        if stop_event.is_set():
            raise RuntimeError("Track fetch requested during shutdown.")
        future = asyncio.run_coroutine_threadsafe(get_next_track(), loop)
        try:
            track_info = future.result(timeout=1.0)
            if track_info is None:
                raise RuntimeError("No track info returned by get_next_track.")
            return track_info
        except Exception as e:
            future.cancel()
            raise RuntimeError("Error fetching next track") from e

    def worker_target() -> None:
        try:
            pipeline.stream_tracks(next_track_blocking, should_stop=stop_event.is_set)
        except Exception as e:
            worker_error.append(e)
            stop_event.set()

    worker = Thread(target=worker_target, daemon=True)
    worker.start()
    try:
        while worker.is_alive():
            if worker_error:
                raise worker_error[0]
            await asyncio.sleep(0.5)
        if worker_error:
            raise worker_error[0]
    except asyncio.CancelledError:
        pass
    finally:
        logging.info("Shutting down streamer...")
        stop_event.set()
        worker.join(timeout=2.0)
        if worker.is_alive():
            logging.warning("Pipeline worker did not stop within 2 seconds.")
        pipeline.close()  # This will join the sender threads
