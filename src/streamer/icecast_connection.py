import logging
from typing import Literal

import shout

from streamer.stream_config import StreamConfig
from streamer.stream_mount import StreamMount


class IcecastConnection:
    def __init__(
        self,
        config: StreamConfig,
        mount: StreamMount,
        *,
        fmt: Literal["mp3", "ogg"],
        allow_metadata: bool,
    ) -> None:
        self.mount_name = mount.mount
        conn = shout.Shout()
        conn.host = config.host
        conn.port = config.port
        conn.user = "source"
        conn.password = config.password
        conn.mount = mount.mount
        conn.format = fmt
        conn.protocol = "http"
        conn.public = mount.public
        conn.name = mount.name
        # conn.audio_info = {
        #     shout.SHOUT_AI_BITRATE: str(opus_bitrate_approx) if fmt == '',
        #     shout.SHOUT_AI_SAMPLERATE: str(sample_rate),
        #     shout.SHOUT_AI_CHANNELS: str(channels),
        # }
        # if mount.description:
        #     conn.description = mount.description
        if mount.genre:
            conn.genre = mount.genre
        if mount.url:
            conn.url = mount.url

        logging.info(f"Connecting to Icecast mount {mount.mount}...")
        conn.open()
        self._conn = conn
        self._allow_metadata = allow_metadata

    def send(self, data: bytes) -> None:
        if data:
            try:
                self._conn.send(data)
            except Exception as e:
                logging.error(f"Error sending to {self.mount_name}: {e}")

    def set_title(self, title: str | None) -> None:
        pass
        # if not self._allow_metadata or not title:
        #     return
        # try:
        #     metadata = shout.Metadata()
        #     metadata.set("title", title)
        #     self._conn.metadata = metadata
        # except Exception as e:
        #     logging.warning(f"Failed to set metadata: {e}")

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass
