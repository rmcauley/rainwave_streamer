import argparse
import asyncio
import logging
import sys
from typing import Sequence

from streamer.encoder_senders.subprocess_encoder import SubprocessEncoderSender
from streamer.sinks.icecast_sink import IcecastSink
from streamer.sinks.null_sink import NullSink
from streamer.stream_config import StreamConfig
from stream_forever import stream_forever

import config
from streamer.track_decoders.gstreamer_track_decoder import GstreamerTrackDecoder


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rainwave Python-native Icecast streamer."
    )
    parser.add_argument("--sid", required=True, type=int, help="Rainwave station ID")
    parser.add_argument(
        "--perftest",
        required=False,
        type=bool,
        help="Run a performance test",
        default=False,
        const=True,
    )
    return parser.parse_args(argv)


def _build_null_stream_config() -> StreamConfig:
    return StreamConfig(
        description="Local sink test; no Icecast connection.",
        genre="Test",
        host="127.0.0.1",
        name="Null Sink Memory Leak Test",
        password="unused",
        port=8000,
        stream_filename="null_sink",
        url="http://localhost/null_sink",
    )


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    args = _parse_args(argv)
    sid: int | None = args.sid
    if sid is None:
        raise Exception("sid argument must be provided")
    station_config = config.stations.get(sid)
    if station_config is None:
        raise Exception("No matching sid in config.py")
    relay_config = next(iter(config.relays.values()), None)
    if relay_config is None:
        raise Exception("No relay config found in config.py")

    stream_config = (
        _build_null_stream_config()
        if args.perftest
        else StreamConfig(
            host=relay_config["ip_address"],
            port=relay_config["port"],
            password=relay_config["source_password"],
            stream_filename=station_config["stream_filename"],
            name=station_config["name"],
            description=station_config["description"],
            genre="Game",
            url=f"http://rainwave.cc/{station_config['stream_filename']}",
        )
    )
    try:
        asyncio.run(
            stream_forever(
                config=stream_config,
                connection=NullSink if args.perftest else IcecastSink,
                decoder=GstreamerTrackDecoder,
                encoder=SubprocessEncoderSender,
                show_memory_usage=args.perftest,
                use_realtime_wait=False if args.perftest else True,
            )
        )
    except KeyboardInterrupt:
        pass
    except Exception:
        logging.exception("Streamer crashed.")
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
