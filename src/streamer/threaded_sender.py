#########################################################################
# This file was AI generated with no human supervision.
#########################################################################

import logging
import queue
from threading import Condition, Event, Thread

from streamer.icecast_connection import IcecastConnection, IcecastConnectionError


class ThreadedSender:
    """
    Decouples audio encoding from network transmission.
    Writes to a queue; background thread reads queue and sends to Icecast.
    """

    def __init__(
        self, conn: IcecastConnection, max_buffer_bytes: int = 64 * 1024
    ) -> None:
        self._conn = conn
        self._queue: queue.Queue[bytes | None] = queue.Queue()
        self._max_buffer_bytes = max_buffer_bytes
        self._buffered_bytes = 0
        self._buffer_cond = Condition()
        self._stop_event = Event()
        self._thread = Thread(
            target=self._worker, daemon=True, name=f"Sender-{conn.mount_name}"
        )
        self._thread.start()

    def _worker(self) -> None:
        while not self._stop_event.is_set():
            data = None
            try:
                data = self._queue.get(timeout=0.25)
                if self._stop_event.is_set():
                    break
                if data is None:
                    break
                self._send_with_retry(data)
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Sender thread error: {e}")
            finally:
                if data is not None:
                    with self._buffer_cond:
                        self._buffered_bytes = max(0, self._buffered_bytes - len(data))
                        self._buffer_cond.notify_all()

    def _send_with_retry(self, data: bytes) -> None:
        # Review note: this streamer is intended to run as a daemon and retry forever.
        # We intentionally keep retrying send/reconnect until shutdown is requested.
        # IcecastConnection.reconnect() may block while network/server is unavailable.
        retry_delay_seconds = 2.0
        while not self._stop_event.is_set():
            try:
                self._conn.send(data)
                return
            except IcecastConnectionError as send_error:
                logging.warning(
                    "Send failed to %s, retrying: %s",
                    self._conn.mount_name,
                    send_error,
                )
                if self._stop_event.wait(timeout=retry_delay_seconds):
                    return
                try:
                    self._conn.reconnect()
                    logging.info(
                        "Reconnected to Icecast mount %s.", self._conn.mount_name
                    )
                except IcecastConnectionError as reconnect_error:
                    logging.warning(
                        "Reconnect failed for %s, will retry: %s",
                        self._conn.mount_name,
                        reconnect_error,
                    )

    def write(self, data: bytes) -> int:
        if not data:
            return 0
        if self._stop_event.is_set():
            return len(data)

        data_len = len(data)
        with self._buffer_cond:
            while not self._stop_event.is_set():
                would_overflow = self._buffered_bytes + data_len > self._max_buffer_bytes
                # Allow one oversized packet through only when buffer is empty.
                oversize_allowed = (
                    data_len > self._max_buffer_bytes and self._buffered_bytes == 0
                )
                if not would_overflow or oversize_allowed:
                    break
                self._buffer_cond.wait(timeout=0.25)

            if self._stop_event.is_set():
                return len(data)

            self._buffered_bytes += data_len

        try:
            self._queue.put_nowait(data)
        except Exception:
            with self._buffer_cond:
                self._buffered_bytes = max(0, self._buffered_bytes - data_len)
                self._buffer_cond.notify_all()
            raise

        return data_len

    def flush(self) -> None:
        # No-op for file-like compatibility with PyAV.
        return

    def close(self) -> None:
        self._stop_event.set()
        with self._buffer_cond:
            self._buffer_cond.notify_all()

        # Best effort wake-up for blocked get(); queue is intentionally not drained on shutdown.
        self._queue.put_nowait(None)

        # Closing the connection unblocks any in-flight network send.
        self._conn.close()

        self._thread.join(timeout=2.0)
        if self._thread.is_alive():
            logging.error("Sender thread did not stop within 2 seconds.")

        # Release any producers blocked on buffer accounting.
        with self._buffer_cond:
            self._buffered_bytes = 0
            self._buffer_cond.notify_all()

    # File-like interface for PyAV
    def readable(self) -> bool:
        return False

    def writable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return False
