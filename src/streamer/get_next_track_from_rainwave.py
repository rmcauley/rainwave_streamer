#########################################################################
# This file was human written.
#########################################################################

from streamer.audio_track import AudioTrackInfo
from collections.abc import Callable

type GetNextTrackFromRainwaveBlockingFn = Callable[[], AudioTrackInfo]
type MarkTrackInvalidOnRainwaveFireAndForgetFn = Callable[[str], None]


async def get_next_track_from_rainwave() -> AudioTrackInfo:
    return AudioTrackInfo(
        path="/mnt/e/Music - VGM/Sonic R OST/003 - Back In Time.mp3",
        gain_db=-6.27,
    )


async def mark_track_invalid_on_rainwave(path: str) -> None:
    pass
