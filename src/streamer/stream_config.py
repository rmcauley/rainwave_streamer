import math
from dataclasses import dataclass
from typing import Callable, Literal


SupportedFormats = Literal["mp3", "ogg"]
ShouldStopFn = Callable[[], bool]

mp3_bitrate_approx = 128
opus_bitrate_approx = 112
sample_rate = 48000
channels = 2
layout = "stereo"

# Length of crossfade, used to make sure our buffer sizes for samples
# are at least this long.
crossfade_seconds = 5
# Keep decoder lookahead at the crossfade window so we don't introduce
# extra app-level buffering beyond what crossfade logic needs.
lookahead_seconds = crossfade_seconds
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
    sid: int
