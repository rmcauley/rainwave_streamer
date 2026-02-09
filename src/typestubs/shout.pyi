from typing import Dict, Optional

SHOUT_AI_BITRATE: int
SHOUT_AI_SAMPLERATE: int
SHOUT_AI_CHANNELS: int

class Metadata:
    def __init__(self) -> None: ...
    def set(self, key: str, value: str) -> None: ...

class Shout:
    host: str
    port: int
    user: str
    password: str
    mount: str
    format: str
    protocol: str
    public: int
    name: str
    description: str
    genre: str
    url: str
    audio_info: Dict[int, str]
    metadata: Optional[Metadata]

    def __init__(self) -> None: ...
    def open(self) -> None: ...
    def send(self, data: bytes) -> None: ...
    def close(self) -> None: ...
