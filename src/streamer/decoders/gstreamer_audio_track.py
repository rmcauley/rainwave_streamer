from collections import deque
import logging
from typing import Any, Iterator

import numpy as np

from streamer.decoders.audio_track import (
    AudioTrack,
    AudioTrackDecodeError,
    AudioTrackNoMoreFramesError,
    AudioTrackOpenError,
)
from streamer.stream_config import (
    sample_rate,
    channels,
    crossfade_seconds,
)

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst


class GstreamerAudioTrack(AudioTrack):
    _pipeline: Any
    _appsink: Any
    _bus: Any
    _eos: bool
    _closed: bool

    def _open_stream(self) -> None:
        self._eos = False

        try:
            Gst.init(None)
            (pipeline, appsink, bus) = self._build_gst_pipeline(self.path)
            self._pipeline = pipeline
            self._appsink = appsink
            self._bus = bus

            state_result = self._pipeline.set_state(Gst.State.PLAYING)
            if state_result == Gst.StateChangeReturn.FAILURE:
                raise AudioTrackOpenError(
                    self.path, "Failed to set GStreamer pipeline to PLAYING."
                )
        except Exception as e:
            self.close()
            logging.error(f"Failed to open track {self.path}: {e}")
            raise AudioTrackOpenError(self.path) from e

    def _make_gst_element(self, element_name: str, instance_name: str) -> Any:
        element = Gst.ElementFactory.make(element_name, instance_name)
        if element is None:
            raise AudioTrackOpenError(
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
        capsfilter = self._make_gst_element("capsfilter", "caps")
        appsink = self._make_gst_element("appsink", "sink")

        filesrc.set_property("location", path)
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
        pipeline.add(capsfilter)
        pipeline.add(appsink)

        if not filesrc.link(decodebin):
            raise AudioTrackOpenError(self.path, "Failed to link filesrc -> decodebin.")
        if not audioconvert.link(audioresample):
            raise AudioTrackOpenError(
                self.path, "Failed to link audioconvert -> audioresample."
            )
        if not audioresample.link(capsfilter):
            raise AudioTrackOpenError(
                self.path, "Failed to link audioresample -> capsfilter."
            )
        if not capsfilter.link(appsink):
            raise AudioTrackOpenError(
                self.path, "Failed to link capsfilter -> appsink."
            )

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

    def get_start_buffer(self) -> deque[np.ndarray]:
        # Get the start-of-song buffer, trimming silence, up to 5 seconds.
        start_buffer: deque[np.ndarray] = deque()
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
                    start_buffer_samples += frame.shape[1]
        except Exception as e:
            logging.error(
                "Failed to decode initial crossfade audio buffer for track %s: %s. Is the song too short to be used on Rainwave?",
                self.path,
                e,
            )
            raise AudioTrackDecodeError(self.path) from e

        return start_buffer

    def _poll_gst_messages(self) -> None:
        if self._bus is None:
            return

        message_types = Gst.MessageType.ERROR | Gst.MessageType.EOS
        while True:
            msg = self._bus.timed_pop_filtered(0, message_types)
            if msg is None:
                return

            if msg.type == Gst.MessageType.ERROR:
                err, debug = msg.parse_error()
                raise AudioTrackDecodeError(
                    self.path, f"GStreamer decode error: {err} (debug: {debug})"
                )
            if msg.type == Gst.MessageType.EOS:
                self._eos = True

    def _sample_to_frame(self, sample: Any) -> np.ndarray:
        buffer = sample.get_buffer()
        if buffer is None:
            return np.empty((channels, 0), dtype=np.float32)

        success, map_info = buffer.map(Gst.MapFlags.READ)
        if not success:
            raise AudioTrackDecodeError(self.path, "Failed to map decoded PCM buffer.")

        try:
            interleaved = np.frombuffer(map_info.data, dtype=np.float32)
            if interleaved.size <= 0:
                return np.empty((channels, 0), dtype=np.float32)

            total_frames = interleaved.size // channels
            if total_frames <= 0:
                return np.empty((channels, 0), dtype=np.float32)

            usable_samples = total_frames * channels
            if usable_samples != interleaved.size:
                interleaved = interleaved[:usable_samples]

            frame = interleaved.reshape(total_frames, channels).T.copy()
            if self._linear_gain != np.float32(1.0):
                np.multiply(frame, self._linear_gain, out=frame, casting="unsafe")
            return frame
        finally:
            buffer.unmap(map_info)

    def _get_resampled_and_gained_next_frames_from_track(self) -> Iterator[np.ndarray]:
        while True:
            self._poll_gst_messages()
            if self._eos:
                raise AudioTrackNoMoreFramesError()

            sample = self._appsink.emit("pull-sample")
            if sample is None:
                self._poll_gst_messages()
                raise AudioTrackNoMoreFramesError()

            frame = self._sample_to_frame(sample)
            if frame.shape[1] <= 0:
                continue
            yield frame
            return
