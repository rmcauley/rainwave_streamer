#########################################################################
# This file was human-written.
#########################################################################

from collections import deque
from dataclasses import dataclass
import logging
import math
from typing import Iterator

import av
from av import AudioFrame, AudioStream
from av.error import EOFError
from av.audio.resampler import AudioResampler
from av.container import InputContainer
import numpy as np

from streamer.stream_constants import sample_rate, layout

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

    # Decoded audio buffer so we can easily and safely trim silence.
    # Will be kept at approx lookahead_seconds length.
    audio_buffer: deque[np.ndarray]
    audio_buffer_samples: int

    def __init__(self, track_info: AudioTrackInfo) -> None:
        logging.info(f"Opening track: {track_info.path}")
        self.path = track_info.path
        self._gain_db = track_info.gain_db

        # AudioResampler is required to maintain 48kHz for Ogg Opus.
        # It also maintains internal state and so must be instantiated
        # per track.
        self._resampler = AudioResampler(
            format="fltp",
            layout=layout,
            rate=sample_rate,
        )

        self.audio_buffer = deque()
        self.audio_buffer_samples = 0

        try:
            self._container = av.open(track_info.path)
            self._stream = self._container.streams.audio[0]
            self._decoder = self._container.decode(self._stream)
        except Exception as e:
            logging.error(f"Failed to open track {track_info.path}: {e}")
            raise

    def get_start_crossfade_buffer(self) -> deque[np.ndarray]:
        # Get the start-of-song buffer, trimming silence, up to 5 seconds.
        start_buffer = deque()
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
            raise

        return start_buffer

    def _get_resampled_frames(self, frame: AudioFrame | None) -> Iterator[np.ndarray]:
        # AudioResampler can yield multiple new AudioFrames per input frame, so we must loop.
        for resampled_frame in self._resampler.resample(frame):
            # We only need to process this frame if it has more than 0 samples in it
            if resampled_frame.samples > 0:
                # This business is our replaygain application.
                yield resampled_frame.to_ndarray() * math.pow(
                    10.0, self._gain_db / 20.0
                )

    # ffmpeg calls encoded chunks "packets"; here we work with decoded frames.
    def _decode_next_frame(self) -> Iterator[np.ndarray]:
        # Get the next decoded frame from pyav.  Will throw StopIteration or EOFError
        # when the file has reached its end.
        decoded_frame = next(self._decoder)
        # AudioResampler can yield multiple new AudioFrames per input frame, so we must loop.
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
        # This try block is for when StopIteration or EOFError has occurred.  Then we know
        # we have reached the end of the MP3.
        try:
            # Fill in the main buffer to the number of seconds required by the lookahead.
            while self.audio_buffer_samples < (lookahead_seconds * sample_rate):
                for frame in self._decode_next_frame():
                    self.audio_buffer.append(frame)
                    self.audio_buffer_samples += frame.shape[1]

            # Now loop until the end of the song, upon which pyav will throw a StopIteration or EOFError.
            while True:
                for frame in self._decode_next_frame():
                    # For each frame we add to the end of buffer...
                    self.audio_buffer.append(frame)
                    self.audio_buffer_samples += frame.shape[1]
                    # ... pop out the first frame.  This is a fast operation thanks to the deque collection.
                    yield_frame = self.audio_buffer.popleft()
                    self.audio_buffer_samples -= yield_frame.shape[1]
                    yield yield_frame
        except (StopIteration, EOFError):
            logging.debug(f"Finished decoding {self.path}")
        except Exception as e:
            logging.error(f"Error decoding {self.path}: {e}")
            raise
        finally:
            self._container.close()

        # Drain the resampler frame queue by using None to flush the AudioResampler
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
