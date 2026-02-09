from dataclasses import dataclass


@dataclass(frozen=True)
class StreamMount:
    mount: str
    bitrate: int
    name: str
    genre: str | None
    url: str | None
    public: int
