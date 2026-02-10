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

# Keep a sample buffer size from audio files of 7.5 seconds.
# This accounts for 5 seconds of crossfade and 2.5 seconds of trimmed silence
# or I/O or encoder delay.
sample_buffer_size = int(7.5 * sample_rate)

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
    _path: str
    _title: str
    _gain_db: float

    # pyav data streams
    _container: InputContainer
    _stream: AudioStream
    _decoder: Iterator[AudioFrame]

    # Decoded audio buffer so we can easily and safely trim silence.
    # Will be kept at approximately the length of sample_buffer_size.
    audio_buffer: deque[np.ndarray]

    def __init__(self, track_info: AudioTrackInfo) -> None:
        logging.info(f"Opening track: {track_info.path}")
        try:
            self._container = av.open(track_info.path)
        except Exception as e:
            logging.error(f"Failed to open track {track_info.path}: {e}")
            raise

        self._path = track_info.path
        self._gain_db = track_info.gain_db

        # AudioResampler is required to maintain 48kHz for Ogg Opus.
        # It also maintains internal state and so must be instantiated
        # per track.
        self._resampler = AudioResampler(
            format="fltp",
            layout=layout,
            rate=sample_rate,
        )

        # Buffer to make silence and crossfading easier for audio pipeline.
        self.audio_buffer = deque()

        # The (presumably) MP3 stream coming from pyav
        self._stream = self._container.streams.audio[0]
        # A single iterator to run over each decoded frame from pyav.
        # Do not use multiple iterators/decoders or you wind up decoding twice
        # and interleaving multiple decoded copies of the track.
        self._decoder = self._container.decode(self._stream)

        # Feed our initial ~7.5s of audio into our audio buffer while trimming silence.
        trimming = True
        try:
            buffered_samples = 0
            while buffered_samples < sample_buffer_size:
                for frame in self._decode_next_frame():
                    # If we are still trimming silence...
                    if trimming:
                        # Check the audio level of this frame
                        if np.max(np.abs(frame)) > silence_threshold_linear:
                            # We have found audio and can stop trimming silence.
                            trimming = False
                        else:
                            # The frame is silent and we can keep decoding and trimming.
                            continue
                    self.audio_buffer.append(frame)
                    buffered_samples += frame.shape[1]
                    # Not adding an explicit 'break' on this inner loop
                    # if the desired sample_buffer_size is reached is intentional.
                    # It may result in adding a little more frames to the buffer
                    # than we intend, but the majority of our source files are
                    # 44.1KHz or 48KHz so this won't result in a major memory overrun.
                    #
                    # It also ensures that all frames that are read from _decode_next_frame
                    # do wind up in the buffer, avoiding any potentially dropped frames.
        except Exception as e:
            logging.error(
                "Failed to decode initial audio buffer for track %s: %s.  Is the song under 10 seconds?",
                self._path,
                e,
            )
            raise
        finally:
            self._container.close()

    def _get_resampled_frames(self, frame: AudioFrame | None) -> Iterator[np.ndarray]:
        # AudioResampler can yield multiple new AudioFrames per input frame, so we must loop.
        for resampled_frame in self._resampler.resample(frame):
            # We only need to process this frame if it has more than 0 samples in it
            if resampled_frame.samples > 0:
                # This business is our replaygain application.
                yield resampled_frame.to_ndarray() * math.pow(
                    10.0, self._gain_db / 20.0
                )

    def _decode_next_frame(self) -> Iterator[np.ndarray]:
        # Get the next frame from pyav.  Will throw StopIteration or EOFError
        # when the file has reached its end.
        decoded_frame = next(self._decoder)
        # AudioResampler can yield multiple new AudioFrames per input frame, so we must loop.
        for resampled_frame in self._get_resampled_frames(decoded_frame):
            yield resampled_frame

    def get_samples(self) -> Iterator[np.ndarray]:
        try:
            # The intent here is to loop until the end of the song, upon which pyav
            # will throw a StopIteration or EOFError.
            while True:
                for frame in self._decode_next_frame():
                    # For each frame we add to the end of buffer...
                    self.audio_buffer.append(frame)
                    # ... pop out the first frame.  This is a fast operation thanks to the deque collection.
                    yield self.audio_buffer.popleft()
        except (StopIteration, EOFError):
            logging.debug(f"Finished decoding {self._path}")
        except Exception as e:
            logging.error(f"Error decoding {self._path}: {e}")
            raise
        finally:
            self._container.close()

        # Drain the resampler frame queue by using None to flush the AudioResampler
        for frame in self._get_resampled_frames(None):
            self.audio_buffer.append(frame)

        # Once StopIteration or EOFError has been thrown in the above loop, we can trim remaining silence:
        self._trim_trailing_silence()

        # Data is purposefully left in the buffer at this point for the pipeline to use for crossfading.
        raise AudioTrackEOFError()

    #########################################################################
    # Beware: below is AI slop territory
    #########################################################################

    def _trim_trailing_silence(self) -> None:
        max_trim_samples = int(2.5 * sample_rate)
        if max_trim_samples <= 0 or not self.audio_buffer:
            return

        trimmed = 0

        while self.audio_buffer and trimmed < max_trim_samples:
            frame = self.audio_buffer[-1]
            frame_samples = frame.shape[1]
            if frame_samples == 0:
                self.audio_buffer.pop()
                continue

            remaining = max_trim_samples - trimmed
            if frame_samples <= remaining:
                amplitudes = np.max(np.abs(frame), axis=0)
                audible_indices = np.where(amplitudes > silence_threshold_linear)[0]
                if len(audible_indices) == 0:
                    trimmed += frame_samples
                    self.audio_buffer.pop()
                    continue
                last_audible = int(audible_indices[-1])
                if last_audible < frame_samples - 1:
                    trimmed += frame_samples - (last_audible + 1)
                    self.audio_buffer[-1] = frame[:, : last_audible + 1]
                return

            tail = frame[:, frame_samples - remaining :]
            amplitudes = np.max(np.abs(tail), axis=0)
            audible_indices = np.where(amplitudes > silence_threshold_linear)[0]
            if len(audible_indices) == 0:
                trimmed += remaining
                self.audio_buffer[-1] = frame[:, : frame_samples - remaining]
                return
            last_audible = int(audible_indices[-1])
            if last_audible < remaining - 1:
                trimmed += remaining - (last_audible + 1)
                self.audio_buffer[-1] = frame[
                    :, : frame_samples - (remaining - (last_audible + 1))
                ]
            return
