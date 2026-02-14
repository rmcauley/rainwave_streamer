from abc import abstractmethod
from typing import Any

from streamer.sinks.sink import (
    AudioSink,
    AudioSinkConstructor,
)
from streamer.stream_config import ShouldStopFn, StreamConfig, SupportedFormats


class EncoderSenderError(Exception):
    pass


class EncoderSenderEncodeError(Exception):
    pass


class EncoderSenderSendError(Exception):
    pass


class EncoderSender:
    _config: StreamConfig
    _format: SupportedFormats
    _conn: AudioSink
    _should_stop: ShouldStopFn

    def __init__(
        self,
        config: StreamConfig,
        format: SupportedFormats,
        connector: AudioSinkConstructor,
        should_stop: ShouldStopFn,
    ) -> None:
        self._config = config
        self._format = format
        self._conn = connector(config, format)
        self._should_stop = should_stop

    @abstractmethod
    def encode_and_send(self, pcm_buffer: Any) -> None:
        raise NotImplementedError()

    def close(self) -> None:
        self._conn.close()


EncoderSenderConstructor = type[EncoderSender]
