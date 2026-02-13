#########################################################################
# This file was primarily human-written, with audio resampling done by AI.
#########################################################################

from collections import deque
from dataclasses import dataclass
import logging
import math
from typing import Iterator

import av
from av import AudioFrame, AudioStream
from av.error import EOFError
from av.container import InputContainer
import numpy as np
import soxr

from streamer.stream_constants import sample_rate, channels

# Length of crossfade, used to make sure our buffer sizes for samples
# are at least this long.
crossfade_seconds = 5
# How long to look ahead to the song to check for silence
lookahead_seconds = 10

# Used to detect silence at the beginning and end of tracks
silence_threshold_linear = math.pow(10.0, -60.0 / 20.0)


# Used to signal to the audio pipeline that the track is finished
# and is ready for crossfade.
class AudioTrackEOFError(Exception):
    pass


# Used internally to signal that ffmpeg has raised StopIteration
class AudioTrackNoMoreFramesError(Exception):
    pass


class AudioTrackOpenError(Exception):
    path: str

    def __init__(self, path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.path = path


class AudioTrackDecodeError(Exception):
    path: str

    def __init__(self, path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.path = path


@dataclass(frozen=True)
class AudioTrackInfo:
    path: str
    gain_db: float


class AudioTrack:
    # This class is designed so that an audio pipeline external to it
    # can iterate over all samples, and the audio pipeline has the
    # track's replaygain applied and silence trimmed for it.
    # It is slightly less efficient than doing it all inside the pipeline,
    # but makes the code easier to read and maintain.

    # Track information
    path: str
    _gain_db: float

    # pyav container (basically the MP3 file itself)
    _container: InputContainer
    # The (presumably) MP3 stream coming from pyav from within the container
    _stream: AudioStream
    # A single iterator to run over each decoded frame from pyav.
    # Do not use multiple iterators/decoders or you wind up decoding twice
    # and interleaving multiple decoded copies of the track.
    _decoder: Iterator[AudioFrame]

    # Required for soxr resampling.
    # I'd use pyav, but pyav/ffmpeg's resampler and graph have memory leaks. :(
    _resampler: soxr.ResampleStream
    _input_channels: int
    _linear_gain: np.float32
    _flush_chunk: np.ndarray

    # Decoded audio buffer so we can easily and safely trim silence.
    # Will be kept at approx lookahead_seconds length.
    audio_buffer: deque[np.ndarray]
    audio_buffer_samples: int

    def __init__(self, track_info: AudioTrackInfo) -> None:
        logging.info(f"Opening track: {track_info.path}")
        self.path = track_info.path
        self._gain_db = track_info.gain_db
        self._linear_gain = np.float32(math.pow(10.0, self._gain_db / 20.0))

        self.audio_buffer = deque()
        self.audio_buffer_samples = 0

        try:
            self._container = av.open(track_info.path)
            self._stream = self._container.streams.audio[0]
            self._decoder = self._container.decode(self._stream)

            self._input_channels = self._stream.codec_context.channels or 0
            if self._input_channels <= 0:
                raise AudioTrackOpenError(self.path, "Invalid source channel count.")

            input_rate_value = (
                self._stream.rate or self._stream.codec_context.sample_rate
            )
            if input_rate_value is None or input_rate_value <= 0:
                raise AudioTrackOpenError(self.path, "Invalid source sample rate.")
            input_rate = float(input_rate_value)
            self._flush_chunk = np.empty((0, self._input_channels), dtype=np.float32)

            self._resampler = soxr.ResampleStream(
                in_rate=input_rate,
                out_rate=float(sample_rate),
                num_channels=self._input_channels,
                dtype="float32",
                quality="HQ",
            )
        except Exception as e:
            logging.error(f"Failed to open track {track_info.path}: {e}")
            if self._container:
                self._container.close()
            raise AudioTrackOpenError(self.path) from e

    def get_start_buffer(self) -> deque[np.ndarray]:
        # Get the start-of-song buffer, trimming silence, up to 5 seconds.
        start_buffer: deque[np.ndarray] = deque()
        start_buffer_samples = 0
        trimming = True
        try:
            while start_buffer_samples < (crossfade_seconds * sample_rate):
                for frame in self._decode_next_frame():
                    # If we are still trimming silence and this frame is silent, skip it.
                    if trimming:
                        if np.max(np.abs(frame)) > silence_threshold_linear:
                            # Set trimming = false but keep processing each frame
                            # so we don't drop any resampled frames.
                            trimming = False
                        else:
                            # Skip to next frame in the loop without appending to buffer
                            # if this is silent.
                            continue

                    start_buffer.append(frame)
                    start_buffer_samples += frame.shape[1]
                    # Do not break at this loop level if we have found non-silent audio.
                    # It may result in adding a little more frames to the buffer
                    # than we intend, but the majority of our source files are
                    # 44.1KHz or 48KHz so this won't result in a major memory overrun.
                    #
                    # More importantly, it also ensures that all frames that are read from _decode_next_frame
                    # do wind up in the buffer, avoiding any potentially dropped frames.
        except Exception as e:
            logging.error(
                "Failed to decode initial crossfade audio buffer for track %s: %s.  Is the song too short to be used on Rainwave?",
                self.path,
                e,
            )
            self._container.close()
            raise AudioTrackDecodeError(self.path) from e

        return start_buffer

    # ffmpeg calls encoded chunks "packets"; here we work with decoded frames.
    def _decode_next_frame(self) -> Iterator[np.ndarray]:
        # Note from AI:
        # _decode_next_frame is a generator, so StopIteration must not escape from it
        # (PEP 479 converts that into RuntimeError). Translate decoder exhaustion into
        # EOFError so callers can handle end-of-track normally.
        try:
            decoded_frame = next(self._decoder)
        except StopIteration as e:
            raise AudioTrackNoMoreFramesError() from e
        # soxr can output variable-length chunks; keep decode/resample boundaries isolated.
        for resampled_frame in self._get_resampled_frames(decoded_frame):
            yield resampled_frame

    def _trim_trailing_silence(self) -> None:
        while self.audio_buffer:
            frame = self.audio_buffer[-1]
            frame_samples = frame.shape[1]
            if frame_samples > 0 and np.max(np.abs(frame)) > silence_threshold_linear:
                return

            popped = self.audio_buffer.pop()
            self.audio_buffer_samples -= popped.shape[1]

    def get_frames(self) -> Iterator[np.ndarray]:
        # This try block is for when EOFError has occurred. Then we know
        # we have reached the end of the MP3.
        try:
            # Fill in the main buffer to the number of seconds required by the lookahead.
            while self.audio_buffer_samples < (lookahead_seconds * sample_rate):
                for frame in self._decode_next_frame():
                    self.audio_buffer.append(frame)
                    self.audio_buffer_samples += frame.shape[1]

            # Now loop until the end of the song, upon which pyav exhaustion is surfaced as EOFError.
            while True:
                for frame in self._decode_next_frame():
                    # For each frame we add to the end of buffer...
                    self.audio_buffer.append(frame)
                    self.audio_buffer_samples += frame.shape[1]
                    # ... pop out the first frame.  This is a fast operation thanks to the deque collection.
                    yield_frame = self.audio_buffer.popleft()
                    self.audio_buffer_samples -= yield_frame.shape[1]
                    yield yield_frame
        except (EOFError, AudioTrackNoMoreFramesError):
            logging.debug(f"Finished decoding {self.path}")
        except Exception as e:
            logging.error(f"Error decoding {self.path}: {e}")
            raise AudioTrackDecodeError(self.path) from e
        finally:
            self._container.close()

        # Drain the soxr queue with a final flush call.
        for frame in self._get_resampled_frames(None):
            self.audio_buffer.append(frame)
            self.audio_buffer_samples += frame.shape[1]

        # Once StopIteration or EOFError has been thrown in the above loop, we can trim remaining silence:
        self._trim_trailing_silence()

        # Now yield frames until we run down to crossfade buffer length, so the pipeline can just take the
        # entirety of the buffer and use it for crossfading.
        while self.audio_buffer_samples > (crossfade_seconds * sample_rate):
            yield_frame = self.audio_buffer.popleft()
            self.audio_buffer_samples -= yield_frame.shape[1]
            yield yield_frame

        # Data is purposefully left in the buffer at this point for the pipeline to use for crossfading.
        raise AudioTrackEOFError()

    #########################################################################
    # Beware: below is AI slop territory
    #########################################################################

    def _to_frames_by_channels(self, frame: AudioFrame) -> np.ndarray:
        # to_ndarray gives planar audio as [channels, samples]; soxr expects [samples, channels].
        array = frame.to_ndarray()
        if array.dtype != np.float32:
            array = array.astype(np.float32, copy=False)
        if array.ndim == 1:
            if self._input_channels != 1:
                raise AudioTrackDecodeError(
                    self.path,
                    f"Unexpected mono frame with {self._input_channels} channels.",
                )
            mono = array.reshape(-1, 1)
            if not mono.flags.c_contiguous:
                return np.ascontiguousarray(mono)
            return mono

        if array.ndim != 2:
            raise AudioTrackDecodeError(
                self.path, f"Unexpected frame shape: {array.shape}"
            )

        if frame.format.is_planar:
            frames_by_channels = array.T
        elif array.shape[1] == self._input_channels:
            frames_by_channels = array
        else:
            raise AudioTrackDecodeError(
                self.path,
                f"Unexpected interleaved frame shape {array.shape} for {self._input_channels} channels.",
            )

        if not frames_by_channels.flags.c_contiguous:
            return np.ascontiguousarray(frames_by_channels)

        return frames_by_channels

    def _match_output_channels(self, array: np.ndarray) -> np.ndarray:
        # Pipeline expects [channels, samples] with stream_constants.channels.
        current_channels = array.shape[1]
        if current_channels == channels:
            return array
        if current_channels == 1 and channels == 2:
            return np.repeat(array, 2, axis=1)
        if current_channels > channels:
            return array[:, :channels]
        pad = np.repeat(array[:, -1:], channels - current_channels, axis=1)
        return np.concatenate([array, pad], axis=1)

    def _get_resampled_frames(self, frame: AudioFrame | None) -> Iterator[np.ndarray]:
        if frame is None:
            resampled = self._resampler.resample_chunk(self._flush_chunk, last=True)
        else:
            source = self._to_frames_by_channels(frame)
            resampled = self._resampler.resample_chunk(source, last=False)

        if resampled.size <= 0:
            return

        if self._linear_gain != np.float32(1.0):
            np.multiply(resampled, self._linear_gain, out=resampled, casting="unsafe")
        resampled = self._match_output_channels(resampled)
        output = resampled.T
        if not output.flags.c_contiguous:
            output = np.ascontiguousarray(output)
        yield output
