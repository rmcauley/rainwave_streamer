import argparse
import asyncio
import logging
import sys
from typing import Sequence

from streamer.stream_config import StreamConfig
from stream_forever import stream_forever

import config


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rainwave Python-native Icecast streamer."
    )
    parser.add_argument("--sid", required=True, type=int, help="Rainwave station ID")
    return parser.parse_args(argv)


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

    stream_config = StreamConfig(
        host=relay_config["ip_address"],
        port=relay_config["port"],
        password=relay_config["source_password"],
        stream_filename=station_config["stream_filename"],
        name=station_config["name"],
        description=station_config["description"],
        genre="Game",
        url=f"http://rainwave.cc/{station_config['stream_filename']}",
    )
    try:
        asyncio.run(stream_forever(stream_config))
    except KeyboardInterrupt:
        pass
    except Exception:
        logging.exception("Streamer crashed.")
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
