from typing import Literal

import av
from av.container import OutputContainer
from av.audio.stream import AudioStream

from streamer.icecast_connection import IcecastConnection
from streamer.threaded_sender import ThreadedSender

mp3_bitrate_approx = 128
opus_bitrate_approx = 128
sample_rate = 48000
channels = 2
layout = "stereo"


class Encoder:
    def __init__(
        self,
        conn: IcecastConnection,
        *,
        codec_name: Literal["mp3", "opus"],
        fmt: Literal["mp3", "ogg"],
    ) -> None:
        self._sender = ThreadedSender(conn)
        # PyAV writes to our ThreadedSender which looks like a file
        self._container = av.open(self._sender, mode="w", format=fmt)

        if codec_name == "mp3":
            self._stream = self.get_mp3_stream(self._container)
        else:
            self._stream = self.get_opus_stream(self._container)

    def get_mp3_stream(self, container: OutputContainer) -> AudioStream:
        stream = container.add_stream(  # pyright: ignore[reportUnknownMemberType]
            "mp3",
            rate=sample_rate,
            options={"global_quality": "8"},
        )
        stream.layout = layout
        return stream

    def get_opus_stream(self, container: OutputContainer) -> AudioStream:
        stream = container.add_stream(  # pyright: ignore[reportUnknownMemberType]
            "libopus", rate=sample_rate, options={"b": "128k", "vbr": "on"}
        )
        stream.layout = layout
        return stream

    def encode(self, frame: av.AudioFrame | None) -> None:
        # Note: frame=None triggers flush in PyAV
        for packet in self._stream.encode(frame):
            self._container.mux(packet)

    def close(self) -> None:
        self.encode(None)  # Flush encoder
        self._container.close()
        self._sender.close()
