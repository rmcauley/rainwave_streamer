import logging
import queue
from threading import Event, Thread

from streamer.icecast_connection import IcecastConnection


class ThreadedSender:
    """
    Decouples audio encoding from network transmission.
    Writes to a queue; background thread reads queue and sends to Icecast.
    """

    def __init__(self, conn: IcecastConnection, buffer_size: int = 500) -> None:
        self._conn = conn
        self._queue: queue.Queue[bytes | None] = queue.Queue(maxsize=buffer_size)
        self._stop_event = Event()
        self._thread = Thread(
            target=self._worker, daemon=True, name=f"Sender-{conn.mount_name}"
        )
        self._thread.start()

    def _worker(self) -> None:
        while not self._stop_event.is_set():
            try:
                data = self._queue.get(timeout=1.0)
                if data is None:
                    break
                self._conn.send(data)
                self._queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Sender thread error: {e}")

    def write(self, data: bytes) -> int:
        try:
            self._queue.put(data, timeout=5.0)  # Backpressure if net is dead
            return len(data)
        except queue.Full:
            logging.warning("Network buffer full! Dropping audio packet.")
            return len(data)

    def flush(self) -> None:
        pass

    def close(self) -> None:
        self._stop_event.set()
        try:
            self._queue.put(None, timeout=1.0)
        except queue.Full:
            pass
        self._thread.join(timeout=2.0)
        self._conn.close()

    # File-like interface for PyAV
    def readable(self) -> bool:
        return False

    def writable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return False
