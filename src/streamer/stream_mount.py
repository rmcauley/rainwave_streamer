from dataclasses import dataclass


@dataclass(frozen=True)
class StreamMount:
    mount: str
    bitrate: int
    name: str
    description: str
    genre: str
    url: str
    public: int
