from abc import abstractmethod

from streamer.stream_config import StreamConfig, SupportedFormats


class AudioSinkError(Exception):
    pass


class AudioSink:
    _config: StreamConfig
    _format: SupportedFormats
    mount_path: str

    def __init__(self, config: StreamConfig, format: SupportedFormats) -> None:
        self._config = config
        self._format = format
        self.mount_path = f"{config.stream_filename}.{format}"

    @abstractmethod
    def send(self, data: bytes) -> None:
        raise NotImplementedError()

    @abstractmethod
    def reconnect(self) -> None:
        raise NotImplementedError()

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError()


AudioSinkConstructor = type[AudioSink]
