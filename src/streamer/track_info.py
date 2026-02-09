from dataclasses import dataclass


@dataclass(frozen=True)
class TrackInfo:
    path: str
    title: str | None = None
    gain_db: float = 0.0
