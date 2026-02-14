from streamer.decoders.audio_track import AudioTrackInfo
from collections.abc import Callable

type GetNextTrackFromRainwaveBlockingFn = Callable[[], AudioTrackInfo]
type MarkTrackInvalidOnRainwaveFireAndForgetFn = Callable[[str], None]

tracks = [
    "104 - Opening Stage.mp3",
    "114 - Boomer Kuwanger Stage.mp3",
    "115 - Sting Chameleon Stage.mp3",
    "117 - Storm Eagle Stage.mp3",
]


async def get_next_track_from_rainwave() -> AudioTrackInfo:
    path = tracks.pop(0)
    tracks.append(path)
    return AudioTrackInfo(
        path=f"/mnt/e/Music - VGM/Mega Man X1 (Boxset OST)/{path}",
        gain_db=-6.27,
    )


async def mark_track_invalid_on_rainwave(path: str) -> None:
    pass
