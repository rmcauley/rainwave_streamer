from dataclasses import dataclass

from streamer.stream_mount import StreamMount


@dataclass(frozen=True)
class StreamConfig:
    host: str
    port: int
    password: str
    mp3: StreamMount
    opus: StreamMount
