from threading import Lock
from typing import Any

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

    def send(self, data: Any) -> None:
        if data is None:
            return
        data_len = int(data.get_size())
        if data_len <= 0:
            return
        with self._lock:
            if self._closed:
                raise AudioSinkError(
                    f"Null sink connection {self.mount_path} is closed."
                )
            self._bytes_sent += data_len

    def close(self) -> None:
        with self._lock:
            self._closed = True
