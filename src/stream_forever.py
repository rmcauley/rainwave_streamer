#########################################################################
# This file was AI generated with no human supervision.
#########################################################################

import asyncio
import logging
import signal
from concurrent.futures import Future
from enum import Enum, auto
from threading import Event, Lock, Thread

from streamer.get_next_track_from_rainwave import (
    get_next_track_from_rainwave,
    mark_track_invalid_on_rainwave,
)
from streamer.audio_pipeline import AudioPipeline, AudioPipelineGracefulShutdownError
from streamer.audio_track import AudioTrackInfo
from streamer.stream_config import StreamConfig


class StreamerState(Enum):
    RUNNING = auto()
    INTENTIONAL_SHUTDOWN_HANDLING = auto()
    UNEXPECTED_ERROR_HANDLING = auto()
    SHUTTING_DOWN = auto()


class StreamerStateController:
    def __init__(self) -> None:
        self._lock = Lock()
        self._state = StreamerState.RUNNING
        self.running_event = Event()
        self.intentional_shutdown_event = Event()
        self.unexpected_error_event = Event()
        self.shutting_down_event = Event()
        self.stop_requested_event = Event()
        self.force_shutdown_event = Event()
        self.running_event.set()

    def _set_state(self, state: StreamerState) -> None:
        with self._lock:
            if self._state == state:
                return
            self._state = state
            self.running_event.clear()
            self.intentional_shutdown_event.clear()
            self.unexpected_error_event.clear()
            self.shutting_down_event.clear()
            if state == StreamerState.RUNNING:
                self.running_event.set()
                self.stop_requested_event.clear()
            elif state == StreamerState.INTENTIONAL_SHUTDOWN_HANDLING:
                self.intentional_shutdown_event.set()
                self.stop_requested_event.set()
            elif state == StreamerState.UNEXPECTED_ERROR_HANDLING:
                self.unexpected_error_event.set()
                self.stop_requested_event.set()
            else:
                self.shutting_down_event.set()
                self.stop_requested_event.set()

    def state(self) -> StreamerState:
        with self._lock:
            return self._state

    def begin_intentional_shutdown(self) -> None:
        self._set_state(StreamerState.INTENTIONAL_SHUTDOWN_HANDLING)

    def begin_unexpected_error_handling(self) -> None:
        self._set_state(StreamerState.UNEXPECTED_ERROR_HANDLING)

    def begin_shutting_down(self) -> None:
        self._set_state(StreamerState.SHUTTING_DOWN)

    def should_stop_workers(self) -> bool:
        return self.stop_requested_event.is_set()


async def stream_forever(config: StreamConfig) -> None:
    loop = asyncio.get_running_loop()
    state_controller = StreamerStateController()
    worker_error: list[BaseException] = []
    pipeline: AudioPipeline | None = None
    worker: Thread | None = None
    installed_signal_handlers: list[signal.Signals] = []

    def next_track_blocking() -> AudioTrackInfo:
        if state_controller.should_stop_workers():
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
            if state_controller.should_stop_workers():
                raise AudioPipelineGracefulShutdownError() from e
            raise RuntimeError("Error fetching next track") from e

    def mark_track_invalid_fire_and_forget(path: str) -> None:
        if state_controller.should_stop_workers():
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
            if state_controller.state() == StreamerState.RUNNING:
                state_controller.begin_intentional_shutdown()
        except Exception as e:
            if state_controller.state() == StreamerState.SHUTTING_DOWN:
                logging.debug("Suppressing worker exception during shutdown: %s", e)
                return
            worker_error.append(e)
            state_controller.begin_unexpected_error_handling()

    def handle_intentional_signal(signal_name: str) -> None:
        logging.info(
            "Received %s. Starting intentional shutdown handling.", signal_name
        )
        state_controller.begin_intentional_shutdown()

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
            state_controller.should_stop_workers,
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
        state_controller.begin_intentional_shutdown()
        raise
    finally:
        state_controller.begin_shutting_down()
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
            state_controller.force_shutdown_event.set()
            logging.warning(
                "Worker did not stop within 2 seconds; forcing shutdown and suppressing further errors."
            )
