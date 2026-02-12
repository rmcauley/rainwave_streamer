#########################################################################
# This file was human written.
#########################################################################

from streamer.audio_track import AudioTrackInfo


async def get_next_track() -> AudioTrackInfo:
    return AudioTrackInfo(
        path="/mnt/e/Music - VGM/Sonic R OST/003 - Back In Time.mp3",
        gain_db=-6.27,
    )
