import math
from dataclasses import dataclass
from typing import Callable, Literal


SupportedFormats = Literal["mp3", "ogg"]
ShouldStopFn = Callable[[], bool]

bitrate_approx = 128
sample_rate = 48000
channels = 2
layout = "stereo"

# Length of crossfade, used to make sure our buffer sizes for samples
# are at least this long.
crossfade_seconds = 5
# How long to look ahead to the song to check for silence
lookahead_seconds = 10
# Used to detect silence at the beginning and end of tracks
silence_threshold_linear = math.pow(10.0, -60.0 / 20.0)


@dataclass(frozen=True)
class StreamConfig:
    description: str
    genre: str
    host: str
    name: str
    password: str
    port: int
    stream_filename: str
    url: str
