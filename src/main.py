import argparse
import asyncio
import logging
import sys
from typing import Sequence

from streamer.encoder_senders.gstreamer_encoder import GstreamerEncoderSender
from streamer.sinks.gstreamer_sink import GstreamerSink
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
        action="store_true",
        help="Run a performance test",
    )
    return parser.parse_args(argv)


def _build_null_stream_config(sid: int) -> StreamConfig:
    return StreamConfig(
        description="Local sink test; no Icecast connection.",
        genre="Test",
        host="127.0.0.1",
        name="Null Sink Memory Leak Test",
        password="unused",
        port=8000,
        stream_filename="null_sink",
        url="http://localhost/null_sink",
        sid=sid,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)

    logging.basicConfig(
        level=logging.ERROR if args.perftest else logging.DEBUG,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

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
        _build_null_stream_config(sid)
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
            sid=sid,
        )
    )
    try:
        asyncio.run(
            stream_forever(
                config=stream_config,
                connection=NullSink if args.perftest else GstreamerSink,
                decoder=GstreamerTrackDecoder,
                encoder=GstreamerEncoderSender,
                show_performance=args.perftest,
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
