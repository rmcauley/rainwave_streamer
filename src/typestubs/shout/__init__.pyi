from __future__ import annotations

PROTOCOL_HTTP: int
FORMAT_MP3: int
FORMAT_OGG: int

class Metadata:
    def __init__(self) -> None: ...
    def set(self, key: str, value: str) -> None: ...

class Shout:
    host: str
    port: int
    user: str
    password: str
    mount: str
    format: int
    protocol: int
    public: int
    name: str
    description: str
    genre: str
    url: str
    metadata: Metadata

    def __init__(self) -> None: ...
    def open(self) -> None: ...
    def close(self) -> None: ...
    def send(self, data: bytes) -> int: ...

__all__: list[str]
