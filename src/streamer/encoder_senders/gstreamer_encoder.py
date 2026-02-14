import logging
from threading import Lock, Thread
from typing import Any

from streamer.encoder_senders.encoder_sender import (
    EncoderSender,
    EncoderSenderEncodeError,
)
from streamer.sinks.sink import AudioSinkConstructor
from streamer.stream_config import (
    ShouldStopFn,
    StreamConfig,
    SupportedFormats,
    mp3_bitrate_approx,
    opus_bitrate_approx,
    channels,
    sample_rate,
)

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst

message_poll_interval_buffers = 16
encoder_buffer_target_ms = 250


class GstreamerEncoderSender(EncoderSender):
    _pipeline: Any
    _appsrc: Any
    _appsink: Any
    _bus: Any
    _appsink_thread: Thread
    _closed: bool
    _close_lock: Lock
    _worker_error: Exception | None
    _worker_error_lock: Lock
    _encode_pushes_since_poll: int
    _appsink_pulls_since_poll: int

    def __init__(
        self,
        config: StreamConfig,
        format: SupportedFormats,
        connector: AudioSinkConstructor,
        should_stop: ShouldStopFn,
    ) -> None:
        super().__init__(config, format, connector, should_stop)
        self._closed = False
        self._close_lock = Lock()
        self._worker_error = None
        self._worker_error_lock = Lock()
        self._encode_pushes_since_poll = 0
        self._appsink_pulls_since_poll = 0

        try:
            Gst.init(None)
            self._pipeline = self._build_pipeline(format)
            self._appsrc = self._get_required_element("src")
            self._appsink = self._get_required_element("sink")
            self._bus = self._pipeline.get_bus()
            if self._bus is None:
                raise EncoderSenderEncodeError("Failed to get GStreamer encoder bus.")

            state_result = self._pipeline.set_state(Gst.State.PLAYING)
            if state_result == Gst.StateChangeReturn.FAILURE:
                raise EncoderSenderEncodeError(
                    "Failed to set GStreamer encoder pipeline to PLAYING."
                )

            self._appsink_thread = Thread(
                target=self._appsink_worker,
                daemon=True,
                name=f"GstEncoderAppsink-{format}-{self._conn.mount_path}",
            )
            self._appsink_thread.start()
        except Exception:
            pipeline = getattr(self, "_pipeline", None)
            if pipeline is not None:
                try:
                    pipeline.set_state(Gst.State.NULL)
                except Exception:
                    pass
            self._conn.close()
            raise

    def encode_and_send(self, pcm_buffer: Any) -> None:
        self._raise_if_worker_failed()

        if pcm_buffer is None:
            return
        if int(pcm_buffer.get_size()) <= 0:
            return

        flow = self._appsrc.emit("push-buffer", pcm_buffer)
        if flow != Gst.FlowReturn.OK:
            self._poll_gst_messages(force=True)
            raise EncoderSenderEncodeError(
                f"GStreamer encoder push-buffer failed: {flow.value_nick}"
            )

        self._encode_pushes_since_poll += 1
        self._poll_gst_messages()
        self._raise_if_worker_failed()

    def close(self) -> None:
        with self._close_lock:
            if self._closed:
                return
            self._closed = True

        appsrc = getattr(self, "_appsrc", None)
        if appsrc is not None:
            try:
                appsrc.emit("end-of-stream")
            except Exception:
                pass

        pipeline = getattr(self, "_pipeline", None)
        if pipeline is not None:
            try:
                pipeline.set_state(Gst.State.NULL)
            except Exception:
                pass

        appsink_thread = getattr(self, "_appsink_thread", None)
        if appsink_thread is not None:
            appsink_thread.join(timeout=2.0)
            if appsink_thread.is_alive():
                logging.error("GStreamer appsink thread did not stop within 2 seconds.")

        self._conn.close()

    def _make_gst_element(self, element_name: str, instance_name: str) -> Any:
        element = Gst.ElementFactory.make(element_name, instance_name)
        if element is None:
            raise EncoderSenderEncodeError(
                f"GStreamer element '{element_name}' was not found. Check plugin installation."
            )
        return element

    def _build_pipeline(self, format: SupportedFormats) -> Any:
        pipeline = Gst.Pipeline.new(f"gstreamer-encoder-{format}")

        appsrc = self._make_gst_element("appsrc", "src")
        audioconvert = self._make_gst_element("audioconvert", "convert")
        appsink = self._make_gst_element("appsink", "sink")

        caps = Gst.Caps.from_string(
            f"audio/x-raw,format=F32LE,channels={channels},rate={sample_rate},layout=interleaved"
        )
        raw_bytes_per_second = sample_rate * channels * 4
        encoder_buffer_target_seconds = encoder_buffer_target_ms / 1000.0
        encoded_kbps = mp3_bitrate_approx if format == "mp3" else opus_bitrate_approx
        encoded_bytes_per_second = int((encoded_kbps * 1000) / 8)
        encoded_max_bytes = max(
            2048, int(encoded_bytes_per_second * encoder_buffer_target_seconds)
        )
        appsrc.set_property("caps", caps)
        appsrc.set_property("is-live", True)
        appsrc.set_property("format", Gst.Format.TIME)
        appsrc.set_property("block", True)
        appsrc.set_property("do-timestamp", True)
        appsrc.set_property(
            "max-bytes", int(raw_bytes_per_second * encoder_buffer_target_seconds)
        )
        appsink.set_property("sync", False)
        appsink.set_property("max-buffers", 0)
        appsink.set_property("max-bytes", encoded_max_bytes)
        appsink.set_property("max-time", int(encoder_buffer_target_ms * 1_000_000))
        appsink.set_property("drop", False)
        appsink.set_property("emit-signals", False)

        pipeline.add(appsrc)
        pipeline.add(audioconvert)
        pipeline.add(appsink)

        if format == "mp3":
            encoder = self._make_gst_element("lamemp3enc", "encoder")
            encoder.set_property("target", "quality")
            encoder.set_property("quality", 7.0)
            encoder.set_property("bitrate", mp3_bitrate_approx)
            pipeline.add(encoder)

            if not appsrc.link(audioconvert):
                raise EncoderSenderEncodeError("Failed to link appsrc -> audioconvert.")
            if not audioconvert.link(encoder):
                raise EncoderSenderEncodeError(
                    "Failed to link audioconvert -> lamemp3enc."
                )
            if not encoder.link(appsink):
                raise EncoderSenderEncodeError("Failed to link lamemp3enc -> appsink.")
            return pipeline

        encoder = self._make_gst_element("opusenc", "encoder")
        muxer = self._make_gst_element("oggmux", "mux")
        encoder.set_property("bitrate-type", "vbr")
        encoder.set_property("bitrate", opus_bitrate_approx * 1000)
        encoder.set_property("audio-type", "generic")
        encoder.set_property("complexity", 10)
        encoder.set_property("bandwidth", "fullband")
        pipeline.add(encoder)
        pipeline.add(muxer)

        if not appsrc.link(audioconvert):
            raise EncoderSenderEncodeError("Failed to link appsrc -> audioconvert.")
        if not audioconvert.link(encoder):
            raise EncoderSenderEncodeError("Failed to link audioconvert -> opusenc.")
        if not encoder.link(muxer):
            raise EncoderSenderEncodeError("Failed to link opusenc -> oggmux.")
        if not muxer.link(appsink):
            raise EncoderSenderEncodeError("Failed to link oggmux -> appsink.")

        return pipeline

    def _get_required_element(self, name: str) -> Any:
        element = self._pipeline.get_by_name(name)
        if element is None:
            raise EncoderSenderEncodeError(
                f"GStreamer encoder pipeline element '{name}' not found."
            )
        return element

    def _appsink_worker(self) -> None:
        try:
            while not self._closed:
                self._poll_gst_messages()
                sample = self._appsink.emit("pull-sample")
                if sample is None:
                    self._poll_gst_messages(force=True)
                    return

                buffer = sample.get_buffer()
                if buffer is None:
                    continue

                self._conn.send(buffer)
                self._appsink_pulls_since_poll += 1
        except Exception as e:
            with self._worker_error_lock:
                self._worker_error = e
            logging.error("GStreamer encoder appsink worker failed: %s", e)

    def _poll_gst_messages(self, *, force: bool = False) -> None:
        bus = self._bus
        if bus is None:
            return
        if not force:
            if self._encode_pushes_since_poll < message_poll_interval_buffers:
                if self._appsink_pulls_since_poll < message_poll_interval_buffers:
                    return
            self._encode_pushes_since_poll = 0
            self._appsink_pulls_since_poll = 0

        message_types = Gst.MessageType.ERROR | Gst.MessageType.EOS
        while True:
            msg = bus.timed_pop_filtered(0, message_types)
            if msg is None:
                return
            if msg.type == Gst.MessageType.ERROR:
                err, debug = msg.parse_error()
                raise EncoderSenderEncodeError(
                    f"GStreamer encode error: {err} (debug: {debug})"
                )
            if msg.type == Gst.MessageType.EOS:
                return

    def _raise_if_worker_failed(self) -> None:
        with self._worker_error_lock:
            worker_error = self._worker_error
        if worker_error is not None:
            raise EncoderSenderEncodeError(
                f"GStreamer encoder worker failed: {worker_error}"
            ) from worker_error
