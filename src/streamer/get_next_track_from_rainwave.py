import logging
from http.client import HTTPConnection
import socket

import cache
import config
from streamer.track_decoders.track_decoder import TrackInfo
from collections.abc import Callable

type GetNextTrackFromRainwaveBlockingFn = Callable[[], TrackInfo]
type MarkTrackInvalidOnRainwaveFireAndForgetFn = Callable[[str], None]


async def get_next_track_from_rainwave(sid: int) -> TrackInfo:
    conn: HTTPConnection | None = None
    track_info: TrackInfo | None = None
    try:
        conn = HTTPConnection("localhost", config.backend_port + int(sid), timeout=2)
        conn.request("GET", "/advance/%s" % sid)
        result = conn.getresponse()
        if result.status == 200:
            liq_annotated_next_song = result.read()
            if not liq_annotated_next_song or len(liq_annotated_next_song) == 0:
                raise Exception("Got zero-length filename from backend!")
            raw_track = liq_annotated_next_song.decode("utf-8").strip()
            if not raw_track.startswith("annotate:"):
                raise Exception("Unexpected backend response format!")

            logging.debug(f"From Rainwave:\n{raw_track}")

            _, _, metadata_and_path = raw_track.partition("annotate:")
            metadata, separator, path = metadata_and_path.rpartition(":")
            if separator == "" or not path:
                raise Exception("Missing track path from backend response!")

            # Safe default till bugs are gone
            gain_db = -10.0
            try:
                replay_gain_marker = 'replay_gain="'
                replay_gain_start = metadata.find(replay_gain_marker)
                if replay_gain_start < 0:
                    raise Exception("Missing replay_gain from backend response!")
                replay_gain_start += len(replay_gain_marker)
                replay_gain_end = metadata.find('"', replay_gain_start)
                if replay_gain_end < 0:
                    raise Exception("Malformed replay_gain in backend response!")

                replay_gain_str = metadata[replay_gain_start:replay_gain_end]
                gain_db = float(replay_gain_str.removesuffix(" dB"))
            except Exception as e:
                logging.error("Could not read replay_gain information.", exc_info=e)
            track_info = TrackInfo(path=path, gain_db=gain_db)
        else:
            raise Exception("HTTP Error %s trying to reach backend!" % result.status)
    except socket.timeout as e:
        await cache.cache_set_station(sid, "backend_ok", False)
        await cache.cache_set_station(sid, "backend_status", repr(e))
        raise
    except Exception as e:
        await cache.cache_set_station(sid, "backend_ok", False)
        await cache.cache_set_station(sid, "backend_status", repr(e))
        raise
    finally:
        if conn:
            conn.close()

    await cache.cache_set_station(sid, "backend_ok", True)
    await cache.cache_set_station(sid, "backend_message", "OK")
    return track_info


async def mark_track_invalid_on_rainwave(path: str) -> None:
    pass


robs_tracks = [
    "/mnt/e/Music - VGM/Mega Man X1 (Boxset OST)/104 - Opening Stage.mp3",
    "/mnt/e/Music - VGM/Mega Man X1 (Boxset OST)/114 - Boomer Kuwanger Stage.mp3",
    "/mnt/e/Music - VGM/Mega Man X1 (Boxset OST)/115 - Sting Chameleon Stage.mp3",
    "/mnt/e/Music - VGM/Mega Man X1 (Boxset OST)/117 - Storm Eagle Stage.mp3",
]


async def get_next_track_from_robs_ssd() -> TrackInfo:
    path = robs_tracks.pop(0)
    robs_tracks.append(path)
    return TrackInfo(
        path=path,
        gain_db=-6.27,
    )
