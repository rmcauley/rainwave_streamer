import logging
import subprocess
from collections import deque
from threading import Lock, Thread
from typing import Any, Literal
from urllib.parse import quote

from streamer.sinks.sink import AudioSinkConstructor
from streamer.sinks.null_sink import NullSink
from streamer.encoder_senders.encoder_sender import (
    EncoderSender,
    EncoderSenderEncodeError,
)
from streamer.stream_config import (
    ShouldStopFn,
    StreamConfig,
    SupportedFormats,
    sample_rate,
    channels,
    mp3_bitrate_approx,
    opus_bitrate_approx,
)
import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst


class FfmpegSubprocessEncoderSender(EncoderSender):
    _codec_name: Literal["mp3", "opus"]
    _process: subprocess.Popen[bytes]
    _stdout_thread: Thread | None
    _stderr_thread: Thread | None
    _pipe_output: bool
    _mount_path: str
    _closed: bool
    _close_lock: Lock
    _stderr_tail: deque[str]
    _worker_error: Exception | None
    _worker_error_lock: Lock
    _conn: Any

    def __init__(
        self,
        config: StreamConfig,
        format: SupportedFormats,
        connector: AudioSinkConstructor,
        should_stop: ShouldStopFn,
    ) -> None:
        self._config = config
        self._format = format
        self._should_stop = should_stop
        self._mount_path = f"{config.stream_filename}.{format}"
        self._codec_name = "mp3" if format == "mp3" else "opus"
        self._pipe_output = connector is NullSink
        self._conn = connector(config, format) if self._pipe_output else None
        self._closed = False
        self._close_lock = Lock()
        self._stderr_tail = deque(maxlen=20)
        self._worker_error = None
        self._worker_error_lock = Lock()
        self._stdout_thread = None
        self._stderr_thread = None
        try:
            self._process = self._start_ffmpeg_process(self._codec_name)
            if self._pipe_output:
                self._stdout_thread = Thread(
                    target=self._stdout_worker,
                    daemon=True,
                    name=f"FfmpegStdout-{format}-{self._mount_path}",
                )
                self._stdout_thread.start()
            self._stderr_thread = Thread(
                target=self._stderr_worker,
                daemon=True,
                name=f"FfmpegStderr-{format}-{self._mount_path}",
            )
            self._stderr_thread.start()
        except Exception:
            process = getattr(self, "_process", None)
            if process is not None:
                try:
                    process.kill()
                except Exception:
                    pass
            if self._conn is not None:
                self._conn.close()
            raise

    def _build_icecast_url(self) -> str:
        host = self._config.host
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"

        mount = self._mount_path.lstrip("/")
        encoded_mount = quote(mount, safe="/")
        return f"icecast://source@{host}:{self._config.port}/{encoded_mount}"

    def _start_ffmpeg_process(
        self, codec_name: Literal["mp3", "opus"]
    ) -> subprocess.Popen[bytes]:
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-f",
            "f32le",
            "-ar",
            str(sample_rate),
            "-ac",
            str(channels),
            "-i",
            "pipe:0",
            "-map_metadata",
            "-1",
        ]

        output_format = "mp3" if codec_name == "mp3" else "ogg"
        content_type = "audio/mpeg" if codec_name == "mp3" else "application/ogg"

        if codec_name == "mp3":
            cmd.extend(
                [
                    "-c:a",
                    "libmp3lame",
                    "-q:a",
                    "7",
                    "-b:a",
                    f"{mp3_bitrate_approx}k",
                ]
            )
        else:
            cmd.extend(
                [
                    "-c:a",
                    "libopus",
                    "-b:a",
                    f"{opus_bitrate_approx}k",
                    "-vbr",
                    "on",
                    "-application",
                    "audio",
                    "-frame_duration",
                    "20",
                    "-compression_level",
                    "10",
                    "-cutoff",
                    "20000",
                ]
            )

        cmd.extend(["-f", output_format])

        if self._pipe_output:
            cmd.append("pipe:1")
        else:
            cmd.extend(
                [
                    "-content_type",
                    content_type,
                    "-ice_name",
                    self._config.name,
                    "-ice_description",
                    self._config.description,
                    "-ice_genre",
                    self._config.genre,
                    "-ice_url",
                    self._config.url,
                    "-ice_public",
                    "0",
                    "-password",
                    self._config.password,
                    self._build_icecast_url(),
                ]
            )

        try:
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE if self._pipe_output else subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                bufsize=0,
            )
        except FileNotFoundError as e:
            raise EncoderSenderEncodeError(
                "ffmpeg executable not found. Is ffmpeg installed and in PATH?"
            ) from e

        if process.stdin is None or process.stderr is None:
            process.kill()
            raise EncoderSenderEncodeError("ffmpeg pipes were not created.")
        if self._pipe_output and process.stdout is None:
            process.kill()
            raise EncoderSenderEncodeError("ffmpeg stdout pipe was not created.")
        return process

    def _set_worker_error(self, error: Exception) -> None:
        with self._worker_error_lock:
            if self._worker_error is None:
                self._worker_error = error

    def _raise_if_worker_failed(self) -> None:
        with self._worker_error_lock:
            worker_error = self._worker_error
        if worker_error is not None:
            raise EncoderSenderEncodeError(
                f"ffmpeg {self._codec_name} worker failed: {worker_error}"
            ) from worker_error

    def _bytes_to_gst_buffer(self, data: bytes) -> Any:
        out_buffer = Gst.Buffer.new_allocate(None, len(data), None)
        if out_buffer is None:
            raise EncoderSenderEncodeError(
                f"Failed to allocate Gst.Buffer for ffmpeg {self._codec_name} output."
            )
        out_buffer.fill(0, data)
        return out_buffer

    def _stdout_worker(self) -> None:
        conn = self._conn
        if conn is None:
            return
        stdout = self._process.stdout
        if stdout is None:
            return
        try:
            while not self._closed:
                data = stdout.read(65536)
                if not data:
                    break
                conn.send(self._bytes_to_gst_buffer(data))
        except Exception as e:
            self._set_worker_error(e)
            logging.error("ffmpeg %s stdout worker failed: %s", self._codec_name, e)

    def _stderr_worker(self) -> None:
        stderr = self._process.stderr
        if stderr is None:
            return
        try:
            while not self._closed:
                line = stderr.readline()
                if not line:
                    break
                text = line.decode(errors="replace").rstrip()
                if text:
                    self._stderr_tail.append(text)
        except Exception as e:
            logging.debug("ffmpeg %s stderr worker failed: %s", self._codec_name, e)

    def _process_error_details(self) -> str:
        if not self._stderr_tail:
            return ""
        return f" ffmpeg stderr: {' | '.join(self._stderr_tail)}"

    def _get_pcm_bytes(self, pcm_buffer: Any) -> bytes:
        if pcm_buffer is None:
            return b""
        if isinstance(pcm_buffer, bytes):
            return pcm_buffer
        if isinstance(pcm_buffer, bytearray):
            return bytes(pcm_buffer)
        if isinstance(pcm_buffer, memoryview):
            return pcm_buffer.tobytes()
        if int(pcm_buffer.get_size()) <= 0:
            return b""
        success, map_info = pcm_buffer.map(Gst.MapFlags.READ)
        if not success:
            raise EncoderSenderEncodeError(
                f"Failed to map PCM Gst.Buffer for ffmpeg {self._codec_name} encoder."
            )
        try:
            return bytes(map_info.data)
        finally:
            pcm_buffer.unmap(map_info)

    def encode_and_send(self, pcm_buffer: Any) -> None:
        self._raise_if_worker_failed()
        pcm_frame_bytes = self._get_pcm_bytes(pcm_buffer)
        if not pcm_frame_bytes:
            return

        if self._process.poll() is not None:
            raise EncoderSenderEncodeError(
                f"ffmpeg {self._codec_name} encoder exited with code {self._process.returncode}."
                f"{self._process_error_details()}"
            )

        stdin = self._process.stdin
        if stdin is None:
            raise EncoderSenderEncodeError(
                f"ffmpeg {self._codec_name} stdin is unavailable."
            )
        try:
            stdin.write(pcm_frame_bytes)
        except BrokenPipeError as e:
            raise EncoderSenderEncodeError(
                f"ffmpeg {self._codec_name} stdin pipe was closed unexpectedly."
                f"{self._process_error_details()}"
            ) from e
        self._raise_if_worker_failed()

    def close(self) -> None:
        with self._close_lock:
            if self._closed:
                return
            self._closed = True

        process = getattr(self, "_process", None)
        if process is None:
            if self._conn is not None:
                self._conn.close()
            return

        try:
            if process.stdin is not None:
                process.stdin.close()
        except Exception:
            pass

        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.kill()
                try:
                    process.wait(timeout=1.0)
                except Exception:
                    pass

        stdout_thread = getattr(self, "_stdout_thread", None)
        if stdout_thread is not None:
            stdout_thread.join(timeout=2.0)
            if stdout_thread.is_alive():
                logging.error("ffmpeg %s stdout thread did not stop.", self._codec_name)

        stderr_thread = getattr(self, "_stderr_thread", None)
        if stderr_thread is not None:
            stderr_thread.join(timeout=2.0)
            if stderr_thread.is_alive():
                logging.error("ffmpeg %s stderr thread did not stop.", self._codec_name)

        if self._conn is not None:
            self._conn.close()
