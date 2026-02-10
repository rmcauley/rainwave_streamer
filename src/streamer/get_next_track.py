from streamer.audio_track import AudioTrackInfo


async def get_next_track() -> AudioTrackInfo:
    """
    Placeholder for caller-supplied async function.
    """
    return AudioTrackInfo(
        path="/mnt/e/Music - VGM/Sonic R OST/003 - Back In Time.mp3",
        title="Rob Test",
        gain_db=-6.27,
    )
