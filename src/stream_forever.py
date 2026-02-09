import asyncio
import logging
from threading import Event, Thread

from streamer.get_next_track import get_next_track
from streamer.audio_pipeline import AudioPipeline
from streamer.stream_config import StreamConfig
from streamer.track_info import TrackInfo


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
        pipeline.close()  # This will join the sender threads
        worker.join(timeout=2.0)
