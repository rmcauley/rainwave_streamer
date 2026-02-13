#########################################################################
# This file was AI generated with close human supervision.
#########################################################################

import logging
from threading import Lock
from typing import Literal

import shout

from streamer.stream_constants import (
    sample_rate,
    channels,
    opus_bitrate_approx,
    mp3_bitrate_approx,
)
from streamer.stream_config import StreamConfig
from streamer.stream_mount import StreamMount


class IcecastConnectionError(Exception):
    pass


class IcecastConnection:
    def __init__(
        self,
        config: StreamConfig,
        mount: StreamMount,
        *,
        fmt: Literal["mp3", "ogg"],
    ) -> None:
        self._config = config
        self._mount = mount
        self._fmt = fmt
        self.mount_name = mount.mount
        self._conn_lock = Lock()
        self._closed = False
        self._conn: shout.Shout | None = self._open_connection()

    def _open_connection(self) -> shout.Shout:
        mount = self._mount
        config = self._config
        fmt = self._fmt

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
        conn.description = mount.description
        conn.genre = mount.genre
        conn.url = mount.url
        conn.audio_info = {
            shout.SHOUT_AI_BITRATE: (
                str(opus_bitrate_approx) if fmt == "ogg" else str(mp3_bitrate_approx)
            ),
            shout.SHOUT_AI_SAMPLERATE: str(sample_rate),
            shout.SHOUT_AI_CHANNELS: str(channels),
        }

        logging.info(f"Connecting to Icecast mount {mount.mount}...")
        try:
            conn.open()
        except Exception as e:
            raise IcecastConnectionError(
                f"Failed opening Icecast connection for {mount.mount}: {e}"
            ) from e
        return conn

    def send(self, data: bytes) -> None:
        if not data:
            return
        with self._conn_lock:
            conn = self._conn
        if conn is None:
            raise RuntimeError(f"Icecast connection {self.mount_name} is closed.")
        # Review note: send() intentionally uses a snapshot of the active connection.
        # During reconnect, this snapshot can be closed concurrently and produce one
        # transient send failure. ThreadedSender retry/reconnect logic handles this,
        # and occasional loss in that failure window is acceptable for this daemon.
        try:
            conn.send(data)
        except Exception as e:
            raise IcecastConnectionError(
                f"Failed sending packet to {self.mount_name}: {e}"
            ) from e

    def reconnect(self) -> None:
        new_conn = self._open_connection()
        old_conn = None
        close_new_conn = False
        with self._conn_lock:
            if self._closed:
                close_new_conn = True
            else:
                old_conn = self._conn
                self._conn = new_conn
        if close_new_conn:
            try:
                new_conn.close()
            except Exception:
                pass
            return
        if old_conn is not None:
            try:
                old_conn.close()
            except Exception:
                pass

    def close(self) -> None:
        with self._conn_lock:
            if self._closed:
                return
            self._closed = True
            conn = self._conn
            self._conn = None
        if conn is None:
            return
        try:
            conn.close()
        except Exception:
            pass
