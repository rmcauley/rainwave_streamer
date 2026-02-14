from threading import Lock

from streamer.sinks.sink import (
    AudioSink,
    AudioSinkError,
)
from streamer.stream_config import StreamConfig, SupportedFormats


class NullSink(AudioSink):
    """
    AudioServerConnection implementation that discards encoded bytes.
    """

    def __init__(self, config: StreamConfig, format: SupportedFormats) -> None:
        super().__init__(config, format)
        self._closed = False
        self._bytes_sent = 0
        self._lock = Lock()

    def send(self, data: bytes) -> None:
        if not data:
            return
        with self._lock:
            if self._closed:
                raise AudioSinkError(
                    f"Null sink connection {self.mount_path} is closed."
                )
            self._bytes_sent += len(data)

    def reconnect(self) -> None:
        with self._lock:
            if self._closed:
                raise AudioSinkError(
                    f"Null sink connection {self.mount_path} is closed."
                )
            return

    def close(self) -> None:
        with self._lock:
            self._closed = True
