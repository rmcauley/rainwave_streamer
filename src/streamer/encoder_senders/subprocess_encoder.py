import numpy as np
from collections import deque
import logging
from threading import Lock, Thread
import subprocess
from typing import Literal

from streamer.connectors.connection import AudioServerConnectionConstructor
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
    bitrate_approx,
)
from streamer.threaded_sender import ThreadedSender


class SubprocessEncoderSender(EncoderSender):
    _codec_name: Literal["mp3", "opus"]
    _process: subprocess.Popen[bytes]
    _sender: ThreadedSender
    _stdout_thread: Thread
    _stderr_thread: Thread
    _closed: bool
    _close_lock: Lock
    _stderr_tail: deque[str]

    def __init__(
        self,
        config: StreamConfig,
        format: SupportedFormats,
        connector: AudioServerConnectionConstructor,
        should_stop: ShouldStopFn,
    ) -> None:
        super().__init__(config, format, connector, should_stop)

        self._closed = False
        self._close_lock = Lock()
        self._stderr_tail = deque(maxlen=20)
        self._sender = ThreadedSender(self._conn, should_stop=should_stop)
        try:
            self._process = self._start_ffmpeg_process(format)
            self._stdout_thread = Thread(
                target=self._stdout_worker,
                daemon=True,
                name=f"FfmpegStdout-{format}-{self._conn.mount_path}",
            )
            self._stderr_thread = Thread(
                target=self._stderr_worker,
                daemon=True,
                name=f"FfmpegStderr-{format}-{self._conn.mount_path}",
            )
            self._stdout_thread.start()
            self._stderr_thread.start()
        except Exception:
            # Roll back the sender thread/connection if encoder initialization fails.
            self._sender.close()
            raise

    def _start_ffmpeg_process(
        self, codec_name: SupportedFormats
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
        ]

        if codec_name == "mp3":
            cmd.extend(
                [
                    "-c:a",
                    "libmp3lame",
                    "-b:a",
                    f"{bitrate_approx}k",
                    "-f",
                    "mp3",
                    "pipe:1",
                ]
            )
        else:
            cmd.extend(
                [
                    "-c:a",
                    "libopus",
                    "-b:a",
                    f"{bitrate_approx}k",
                    "-vbr",
                    "on",
                    "-application",
                    "audio",
                    "-f",
                    "ogg",
                    "pipe:1",
                ]
            )

        try:
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
            )
        except FileNotFoundError as e:
            raise EncoderSenderEncodeError(
                "ffmpeg executable not found. Is ffmpeg installed and in PATH?"
            ) from e

        if process.stdin is None or process.stdout is None or process.stderr is None:
            process.kill()
            raise EncoderSenderEncodeError("ffmpeg pipes were not created.")
        return process

    def _stdout_worker(self) -> None:
        stdout = self._process.stdout
        if stdout is None:
            return
        try:
            while True:
                data = stdout.read(65536)
                if not data:
                    break
                self._sender.write(data)
        except Exception as e:
            logging.error("ffmpeg %s stdout worker failed: %s", self._codec_name, e)

    def _stderr_worker(self) -> None:
        stderr = self._process.stderr
        if stderr is None:
            return
        try:
            while True:
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

    def encode_and_send(self, np_frame: np.ndarray) -> None:
        if np_frame.dtype != np.float32:
            np_frame = np_frame.astype(np.float32, copy=False)
        interleaved = np_frame.T
        if not interleaved.flags.c_contiguous:
            interleaved = np.ascontiguousarray(interleaved)
        pcm_frame_bytes = interleaved.tobytes()

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

    def close(self) -> None:
        super().close()

        with self._close_lock:
            if self._closed:
                return
            self._closed = True

        process = self._process
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

        self._stdout_thread.join(timeout=2.0)
        self._stderr_thread.join(timeout=2.0)
        self._sender.close()
