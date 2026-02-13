#########################################################################
# This file was AI generated with some human supervision.
#########################################################################

import logging
from threading import Event, Lock, Thread
from typing import Literal

import shout

from streamer.stream_constants import (
    sample_rate,
    channels,
    opus_bitrate_approx,
    mp3_bitrate_approx,
)
from streamer.stream_config import StreamConfig
from streamer.stream_mount import StreamMount


class IcecastConnectionError(Exception):
    pass


class IcecastConnection:
    def __init__(
        self,
        config: StreamConfig,
        mount: StreamMount,
        *,
        fmt: Literal["mp3", "ogg"],
    ) -> None:
        self._config = config
        self._mount = mount
        self._fmt = fmt
        self.mount_name = mount.mount
        self._conn_lock = Lock()
        self._reconnect_lock = Lock()
        self._closed = False
        self._closed_event = Event()
        self._reconnect_done = Event()
        self._reconnect_thread: Thread | None = None
        self._reconnect_error: Exception | None = None
        self._conn: shout.Shout | None = self._open_connection()

    def _open_connection(self) -> shout.Shout:
        mount = self._mount
        config = self._config
        fmt = self._fmt

        conn = shout.Shout()
        conn.host = config.host
        conn.port = config.port
        conn.user = "source"
        conn.password = config.password
        conn.mount = mount.mount
        conn.format = fmt
        conn.protocol = "http"
        conn.public = mount.public
        conn.name = mount.name
        conn.description = mount.description
        conn.genre = mount.genre
        conn.url = mount.url
        conn.audio_info = {
            shout.SHOUT_AI_BITRATE: (
                str(opus_bitrate_approx) if fmt == "ogg" else str(mp3_bitrate_approx)
            ),
            shout.SHOUT_AI_SAMPLERATE: str(sample_rate),
            shout.SHOUT_AI_CHANNELS: str(channels),
        }

        logging.info(f"Connecting to Icecast mount {mount.mount}...")
        try:
            conn.open()
        except Exception as e:
            raise IcecastConnectionError(
                f"Failed opening Icecast connection for {mount.mount}: {e}"
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
                raise IcecastConnectionError(
                    f"Icecast connection {self.mount_name} is closed."
                )

            try:
                conn.send(data)
                return
            except Exception as send_error:
                if self._closed_event.is_set():
                    raise IcecastConnectionError(
                        f"Failed sending packet to {self.mount_name}: connection closed."
                    ) from send_error
                logging.warning(
                    "Send failed to %s, reconnecting: %s",
                    self.mount_name,
                    send_error,
                )
                try:
                    if not self._reconnect_with_shutdown_support():
                        raise IcecastConnectionError(
                            f"Failed sending packet to {self.mount_name}: connection closed."
                        ) from send_error
                    logging.info("Reconnected to Icecast mount %s.", self.mount_name)
                    # Re-attempt send immediately after a successful reconnect.
                    continue
                except IcecastConnectionError as reconnect_error:
                    logging.warning(
                        "Reconnect failed for %s, will retry: %s",
                        self.mount_name,
                        reconnect_error,
                    )
                if self._closed_event.wait(timeout=retry_delay_seconds):
                    raise IcecastConnectionError(
                        f"Failed sending packet to {self.mount_name}: connection closed."
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
                    name=f"IcecastReconnect-{self.mount_name}",
                )
                self._reconnect_thread = reconnect_thread
                reconnect_thread.start()

        while not self._reconnect_done.wait(timeout=0.25):
            if self._closed_event.is_set():
                return False

        with self._reconnect_lock:
            reconnect_error = self._reconnect_error

        if reconnect_error is not None:
            if isinstance(reconnect_error, IcecastConnectionError):
                raise reconnect_error
            raise IcecastConnectionError(
                f"Unexpected reconnect error for {self.mount_name}: {reconnect_error}"
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
                    self.mount_name,
                )
