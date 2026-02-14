import numpy as np
from streamer.sinks.sink import (
    AudioSinkConstructor,
)
from streamer.encoder_senders.encoder_sender import EncoderSender
from streamer.stream_config import ShouldStopFn, StreamConfig, SupportedFormats


class GstreamerEncoderSender(EncoderSender):
    def __init__(
        self,
        config: StreamConfig,
        format: SupportedFormats,
        connector: AudioSinkConstructor,
        should_stop: ShouldStopFn,
    ) -> None:
        super().__init__(config, format, connector, should_stop)

    def encode_and_send(self, np_frame: np.ndarray) -> None:
        pass

    def close(self) -> None:
        super().close()
