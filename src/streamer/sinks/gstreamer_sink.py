import logging
from threading import Lock
from typing import Any

import gi

from streamer.sinks.sink import AudioSink, AudioSinkError
from streamer.stream_config import StreamConfig, SupportedFormats

gi.require_version("Gst", "1.0")
from gi.repository import Gst

message_poll_interval_buffers = 64


class GstreamerSink(AudioSink):
    _pipeline: Any
    _appsrc: Any
    _bus: Any
    _closed: bool
    _lock: Lock
    _pushes_since_poll: int

    def __init__(self, config: StreamConfig, format: SupportedFormats) -> None:
        super().__init__(config, format)
        self._closed = False
        self._lock = Lock()
        self._pushes_since_poll = 0

        Gst.init(None)
        self._pipeline = self._build_pipeline()
        self._appsrc = self._get_required_element("src")
        self._bus = self._pipeline.get_bus()
        if self._bus is None:
            raise AudioSinkError("Failed to get GStreamer sink bus.")

        state_result = self._pipeline.set_state(Gst.State.PLAYING)
        if state_result == Gst.StateChangeReturn.FAILURE:
            raise AudioSinkError("Failed to set GStreamer sink pipeline to PLAYING.")

    def _make_gst_element(self, element_name: str, instance_name: str) -> Any:
        element = Gst.ElementFactory.make(element_name, instance_name)
        if element is None:
            raise AudioSinkError(
                f"GStreamer element '{element_name}' was not found. Check plugin installation."
            )
        return element

    def _format_caps(self) -> str:
        if self._format == "mp3":
            return "audio/mpeg,mpegversion=(int)1,layer=(int)3"
        return "application/ogg"

    def _build_pipeline(self) -> Any:
        pipeline = Gst.Pipeline.new(f"gstreamer-sink-{self._format}")
        if pipeline is None:
            raise AudioSinkError("Failed to create GStreamer sink pipeline.")

        appsrc = self._make_gst_element("appsrc", "src")
        queue = self._make_gst_element("queue", "queue")
        shout2send = self._make_gst_element("shout2send", "sink")

        caps = Gst.Caps.from_string(self._format_caps())
        appsrc.set_property("caps", caps)
        appsrc.set_property("is-live", True)
        appsrc.set_property("format", Gst.Format.BYTES)
        appsrc.set_property("block", True)

        shout2send.set_property("ip", self._config.host)
        shout2send.set_property("port", self._config.port)
        shout2send.set_property("username", "source")
        shout2send.set_property("password", self._config.password)
        shout2send.set_property("mount", self.mount_path)
        shout2send.set_property("streamname", self._config.name)
        shout2send.set_property("description", self._config.description)
        shout2send.set_property("genre", self._config.genre)
        shout2send.set_property("url", self._config.url)
        shout2send.set_property("public", False)
        shout2send.set_property("protocol", "http")
        shout2send.set_property("sync", False)

        pipeline.add(appsrc)
        pipeline.add(queue)
        pipeline.add(shout2send)

        if not appsrc.link(queue):
            raise AudioSinkError("Failed to link appsrc -> queue for GStreamer sink.")
        if not queue.link(shout2send):
            raise AudioSinkError("Failed to link queue -> shout2send.")

        return pipeline

    def _get_required_element(self, name: str) -> Any:
        element = self._pipeline.get_by_name(name)
        if element is None:
            raise AudioSinkError(f"GStreamer sink pipeline element '{name}' not found.")
        return element

    def _poll_gst_messages(self, *, force: bool = False) -> None:
        bus = self._bus
        if bus is None:
            return
        if not force:
            if self._pushes_since_poll < message_poll_interval_buffers:
                return
            self._pushes_since_poll = 0

        message_types = Gst.MessageType.ERROR | Gst.MessageType.EOS
        while True:
            msg = bus.timed_pop_filtered(0, message_types)
            if msg is None:
                return
            if msg.type == Gst.MessageType.ERROR:
                err, debug = msg.parse_error()
                raise AudioSinkError(
                    f"GStreamer sink error for {self.mount_path}: {err} (debug: {debug})"
                )
            if msg.type == Gst.MessageType.EOS:
                logging.info("GStreamer sink reached EOS for %s.", self.mount_path)
                return

    def send(self, data: Any) -> None:
        if data is None:
            return

        with self._lock:
            if self._closed:
                raise AudioSinkError(f"GStreamer sink {self.mount_path} is closed.")

            if int(data.get_size()) <= 0:
                return

            flow = self._appsrc.emit("push-buffer", data)
            if flow != Gst.FlowReturn.OK:
                self._poll_gst_messages(force=True)
                raise AudioSinkError(
                    f"GStreamer sink push-buffer failed for {self.mount_path}: {flow.value_nick}"
                )
            self._pushes_since_poll += 1
            self._poll_gst_messages()

    def reconnect(self) -> None:
        with self._lock:
            if self._closed:
                raise AudioSinkError(f"GStreamer sink {self.mount_path} is closed.")
            self._pipeline.set_state(Gst.State.NULL)
            state_result = self._pipeline.set_state(Gst.State.PLAYING)
            if state_result == Gst.StateChangeReturn.FAILURE:
                raise AudioSinkError(
                    f"Failed reconnecting GStreamer sink for {self.mount_path}."
                )

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            try:
                self._appsrc.emit("end-of-stream")
            except Exception:
                pass
            try:
                self._pipeline.set_state(Gst.State.NULL)
            except Exception:
                pass
