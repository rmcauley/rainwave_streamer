import logging
from threading import Event, Lock, Thread

import shout

from streamer.sinks.sink import (
    AudioSink,
    AudioSinkError,
)
from streamer.stream_config import StreamConfig, SupportedFormats, sample_rate, channels


class IcecastSink(AudioSink):
    def __init__(self, config: StreamConfig, format: SupportedFormats) -> None:
        super().__init__(config, format)

        self._conn_lock = Lock()
        self._reconnect_lock = Lock()
        self._closed = False
        self._closed_event = Event()
        self._reconnect_done = Event()
        self._reconnect_thread: Thread | None = None
        self._reconnect_error: Exception | None = None
        self._conn: shout.Shout | None = self._open_connection()

    def _open_connection(self) -> shout.Shout:
        config = self._config

        conn = shout.Shout()
        conn.host = config.host
        conn.port = config.port
        conn.user = "source"
        conn.password = config.password
        conn.mount = self.mount_path
        conn.format = self._format
        conn.protocol = "http"
        conn.public = 0
        conn.name = self._config.name
        conn.description = self._config.description
        conn.genre = self._config.genre
        conn.url = self._config.url
        conn.audio_info = {
            shout.SHOUT_AI_BITRATE: "128",
            shout.SHOUT_AI_SAMPLERATE: str(sample_rate),
            shout.SHOUT_AI_CHANNELS: str(channels),
        }

        logging.info(f"Connecting to Icecast mount {self.mount_path}...")
        try:
            conn.open()
        except Exception as e:
            raise AudioSinkError(
                f"Failed opening Icecast connection for {self.mount_path}: {e}"
            ) from e
        return conn

    def send(self, data: bytes) -> None:
        if not data:
            return
        retry_delay_seconds = 2.0
        while True:
            with self._conn_lock:
                conn = self._conn
            if conn is None:
                raise AudioSinkError(f"Icecast connection {self.mount_path} is closed.")

            try:
                conn.send(data)
                return
            except Exception as send_error:
                if self._closed_event.is_set():
                    raise AudioSinkError(
                        f"Failed sending packet to {self.mount_path}: connection closed."
                    ) from send_error
                logging.warning(
                    "Send failed to %s, reconnecting: %s",
                    self.mount_path,
                    send_error,
                )
                try:
                    if not self._reconnect_with_shutdown_support():
                        raise AudioSinkError(
                            f"Failed sending packet to {self.mount_path}: connection closed."
                        ) from send_error
                    logging.info("Reconnected to Icecast mount %s.", self.mount_path)
                    # Re-attempt send immediately after a successful reconnect.
                    continue
                except AudioSinkError as reconnect_error:
                    logging.warning(
                        "Reconnect failed for %s, will retry: %s",
                        self.mount_path,
                        reconnect_error,
                    )
                if self._closed_event.wait(timeout=retry_delay_seconds):
                    raise AudioSinkError(
                        f"Failed sending packet to {self.mount_path}: connection closed."
                    ) from send_error

    def _run_reconnect(self) -> None:
        try:
            self.reconnect()
            with self._reconnect_lock:
                self._reconnect_error = None
        except Exception as e:
            with self._reconnect_lock:
                self._reconnect_error = e
        finally:
            self._reconnect_done.set()

    def _reconnect_with_shutdown_support(self) -> bool:
        with self._reconnect_lock:
            reconnect_thread = self._reconnect_thread
            if reconnect_thread is None or not reconnect_thread.is_alive():
                self._reconnect_error = None
                self._reconnect_done.clear()
                reconnect_thread = Thread(
                    target=self._run_reconnect,
                    daemon=True,
                    name=f"IcecastReconnect-{self.mount_path}",
                )
                self._reconnect_thread = reconnect_thread
                reconnect_thread.start()

        while not self._reconnect_done.wait(timeout=0.25):
            if self._closed_event.is_set():
                return False

        with self._reconnect_lock:
            reconnect_error = self._reconnect_error

        if reconnect_error is not None:
            if isinstance(reconnect_error, AudioSinkError):
                raise reconnect_error
            raise AudioSinkError(
                f"Unexpected reconnect error for {self.mount_path}: {reconnect_error}"
            ) from reconnect_error
        return True

    def reconnect(self) -> None:
        new_conn = self._open_connection()
        old_conn = None
        close_new_conn = False
        with self._conn_lock:
            if self._closed:
                close_new_conn = True
            else:
                old_conn = self._conn
                self._conn = new_conn
        if close_new_conn:
            try:
                new_conn.close()
            except Exception:
                pass
            return
        if old_conn is not None:
            try:
                old_conn.close()
            except Exception:
                pass

    def close(self) -> None:
        with self._conn_lock:
            if self._closed:
                return
            self._closed = True
            self._closed_event.set()
            conn = self._conn
            self._conn = None
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass

        with self._reconnect_lock:
            reconnect_thread = self._reconnect_thread
        if reconnect_thread is not None and reconnect_thread.is_alive():
            # Review note: reconnect can block inside native I/O and cannot be force-cancelled
            # from Python. Process exit will reclaim these resources.
            reconnect_thread.join(timeout=2.0)
            if reconnect_thread.is_alive():
                logging.warning(
                    "Reconnect thread for %s did not stop within 2 seconds.",
                    self.mount_path,
                )
