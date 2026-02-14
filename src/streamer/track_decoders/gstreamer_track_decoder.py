from collections import deque
import logging
from typing import Any, Iterator

from streamer.track_decoders.track_decoder import (
    TrackDecoder,
    TrackFrame,
    TrackDecodeError,
    TrackNoMoreFramesError,
    TrackOpenError,
)
from streamer.stream_config import (
    sample_rate,
    channels,
    crossfade_seconds,
)

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst

message_poll_interval_buffers = 64


class GstreamerTrackDecoder(TrackDecoder):
    _pipeline: Any
    _appsink: Any
    _bus: Any
    _eos: bool
    _closed: bool
    _pulls_since_poll: int

    def _open_stream(self) -> None:
        self._eos = False
        self._pulls_since_poll = 0

        try:
            Gst.init(None)
            (pipeline, appsink, bus) = self._build_gst_pipeline(self.path)
            self._pipeline = pipeline
            self._appsink = appsink
            self._bus = bus
            if bus is None:
                raise TrackOpenError(self.path, "Failed to get GStreamer bus.")

            state_result = self._pipeline.set_state(Gst.State.PLAYING)
            if state_result == Gst.StateChangeReturn.FAILURE:
                raise TrackOpenError(
                    self.path, "Failed to set GStreamer pipeline to PLAYING."
                )
        except Exception as e:
            self.close()
            logging.error(f"Failed to open track {self.path}: {e}")
            raise TrackOpenError(self.path) from e

    def _make_gst_element(self, element_name: str, instance_name: str) -> Any:
        element = Gst.ElementFactory.make(element_name, instance_name)
        if element is None:
            raise TrackOpenError(
                self.path,
                f"GStreamer element '{element_name}' was not found. Check plugin installation.",
            )
        return element

    def _build_gst_pipeline(self, path: str) -> tuple[Any, Any, Any]:
        pipeline = Gst.Pipeline.new("audio-track-decode")

        filesrc = self._make_gst_element("filesrc", "src")
        decodebin = self._make_gst_element("decodebin", "decode")
        audioconvert = self._make_gst_element("audioconvert", "convert")
        audioresample = self._make_gst_element("audioresample", "resample")
        volume = self._make_gst_element("volume", "gain")
        capsfilter = self._make_gst_element("capsfilter", "caps")
        appsink = self._make_gst_element("appsink", "sink")

        filesrc.set_property("location", path)
        volume.set_property("volume", float(self._linear_gain))
        caps = Gst.Caps.from_string(
            f"audio/x-raw,format=F32LE,channels={channels},rate={sample_rate},layout=interleaved"
        )
        capsfilter.set_property("caps", caps)
        appsink.set_property("sync", False)
        appsink.set_property("max-buffers", 8)
        appsink.set_property("drop", False)
        appsink.set_property("emit-signals", False)

        pipeline.add(filesrc)
        pipeline.add(decodebin)
        pipeline.add(audioconvert)
        pipeline.add(audioresample)
        pipeline.add(volume)
        pipeline.add(capsfilter)
        pipeline.add(appsink)

        if not filesrc.link(decodebin):
            raise TrackOpenError(self.path, "Failed to link filesrc -> decodebin.")
        if not audioconvert.link(audioresample):
            raise TrackOpenError(
                self.path, "Failed to link audioconvert -> audioresample."
            )
        if not audioresample.link(volume):
            raise TrackOpenError(self.path, "Failed to link audioresample -> volume.")
        if not volume.link(capsfilter):
            raise TrackOpenError(self.path, "Failed to link volume -> capsfilter.")
        if not capsfilter.link(appsink):
            raise TrackOpenError(self.path, "Failed to link capsfilter -> appsink.")

        decodebin.connect("pad-added", self._on_decodebin_pad_added, audioconvert)
        bus = pipeline.get_bus()

        return (pipeline, appsink, bus)

    def _on_decodebin_pad_added(
        self, decodebin: Any, src_pad: Any, audioconvert: Any
    ) -> None:
        del decodebin
        sink_pad = audioconvert.get_static_pad("sink")
        if sink_pad is None or sink_pad.is_linked():
            return

        caps = src_pad.get_current_caps()
        if caps is None:
            caps = src_pad.query_caps(None)
        if caps is None:
            return
        caps_str = caps.to_string()
        if not caps_str.startswith("audio/"):
            return

        src_pad.link(sink_pad)

    def _close_stream(self) -> None:
        pipeline = getattr(self, "_pipeline", None)
        self._pipeline = None
        self._appsink = None
        self._bus = None
        if pipeline is not None and Gst is not None:
            try:
                pipeline.set_state(Gst.State.NULL)
            except Exception:
                pass

    def _buffer_samples(self, buffer: Any) -> int:
        if buffer is None:
            return 0
        # F32LE interleaved; each sample across all channels is 4 * channels bytes.
        bytes_per_sample = channels * 4
        size = int(buffer.get_size())
        if size <= 0:
            return 0
        return size // bytes_per_sample

    def get_start_buffer(self) -> deque[TrackFrame]:
        # Get the start-of-song buffer, trimming silence, up to 5 seconds.
        start_buffer: deque[TrackFrame] = deque()
        start_buffer_samples = 0
        trimming = True
        try:
            while start_buffer_samples < (crossfade_seconds * sample_rate):
                for frame in self._get_resampled_and_gained_next_frames_from_track():
                    if trimming:
                        if not self._is_frame_silent(frame):
                            trimming = False
                        else:
                            continue

                    start_buffer.append(frame)
                    start_buffer_samples += frame.samples
        except Exception as e:
            logging.error(
                "Failed to decode initial crossfade audio buffer for track %s: %s. Is the song too short to be used on Rainwave?",
                self.path,
                e,
            )
            raise TrackDecodeError(self.path) from e

        return start_buffer

    def _poll_gst_messages(self, *, force: bool = False) -> None:
        bus = self._bus
        if bus is None:
            return
        if not force:
            if self._pulls_since_poll < message_poll_interval_buffers:
                return
            self._pulls_since_poll = 0

        message_types = Gst.MessageType.ERROR | Gst.MessageType.EOS
        while True:
            msg = bus.timed_pop_filtered(0, message_types)
            if msg is None:
                return

            if msg.type == Gst.MessageType.ERROR:
                err, debug = msg.parse_error()
                raise TrackDecodeError(
                    self.path, f"GStreamer decode error: {err} (debug: {debug})"
                )
            if msg.type == Gst.MessageType.EOS:
                self._eos = True

    def _get_resampled_and_gained_next_frames_from_track(self) -> Iterator[TrackFrame]:
        while True:
            self._poll_gst_messages()
            if self._eos:
                raise TrackNoMoreFramesError()

            sample = self._appsink.emit("pull-sample")
            if sample is None:
                self._poll_gst_messages(force=True)
                raise TrackNoMoreFramesError()

            buffer = sample.get_buffer()
            if buffer is None:
                continue

            samples = self._buffer_samples(buffer)
            if samples <= 0:
                continue

            self._pulls_since_poll += 1
            yield TrackFrame(buffer=buffer, samples=samples)
            return
