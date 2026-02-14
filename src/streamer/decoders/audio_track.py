from abc import abstractmethod
from collections import deque
from dataclasses import dataclass
import logging
import math
from typing import Iterator

import numpy as np

from streamer.stream_config import (
    sample_rate,
    crossfade_seconds,
    lookahead_seconds,
    silence_threshold_linear,
)


# Used to signal to the audio pipeline that the track is finished
# and is ready for crossfade.
class AudioTrackEOFError(Exception):
    pass


# Used internally to signal that decode has reached end-of-file.
class AudioTrackNoMoreFramesError(Exception):
    pass


class AudioTrackOpenError(Exception):
    path: str

    def __init__(self, path: str, *args: object):
        super().__init__(*args)
        self.path = path


class AudioTrackDecodeError(Exception):
    path: str

    def __init__(self, path: str, *args: object):
        super().__init__(*args)
        self.path = path


@dataclass(frozen=True)
class AudioTrackInfo:
    path: str
    gain_db: float


class AudioTrack:
    path: str
    _gain_db: float
    _linear_gain: np.float32
    _closed: bool

    # Decoded audio buffer so we can easily and safely trim silence.
    # Will be kept at approx lookahead_seconds length.
    audio_buffer: deque[np.ndarray]
    audio_buffer_samples: int

    def __init__(self, track_info: AudioTrackInfo) -> None:
        logging.info(f"Opening track: {track_info.path}")
        self.path = track_info.path
        self._gain_db = track_info.gain_db
        self._linear_gain = np.float32(math.pow(10.0, self._gain_db / 20.0))
        self._closed = False

        self.audio_buffer = deque()
        self.audio_buffer_samples = 0

        try:
            self._open_stream()
        except Exception as e:
            self.close()
            logging.error(f"Failed to open track {track_info.path}: {e}")
            raise AudioTrackOpenError(self.path) from e

    @abstractmethod
    def _open_stream(self) -> None:
        # Opens a single reader that will be used to get audio frames from the file.
        raise NotImplementedError()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        self._close_stream()

    @abstractmethod
    def _close_stream(self) -> None:
        pass

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _is_frame_silent(self, frame: np.ndarray) -> bool:
        return frame.shape[1] > 0 and np.max(np.abs(frame)) <= silence_threshold_linear

    @abstractmethod
    def _get_resampled_and_gained_next_frames_from_track(self) -> Iterator[np.ndarray]:
        # Decodes the next packet(s) from the track, resamples to `sample_rate`, applies gain,
        # and yields the resulting frame(s).
        raise NotImplementedError()

    @abstractmethod
    def get_start_buffer(self) -> deque[np.ndarray]:
        # Returns the first `crossfade_seconds` from the track reader using _get_resampled_and_gained_next_frames_from_track.
        # Skips all silent frames.
        # Does not use self._audio_buffer, returns a fresh buffer while consuming from a shared reader.
        raise NotImplementedError()

    def _trim_trailing_silence(self) -> None:
        while self.audio_buffer:
            frame = self.audio_buffer[-1]
            if not self._is_frame_silent(frame):
                return

            popped = self.audio_buffer.pop()
            self.audio_buffer_samples -= popped.shape[1]

    def get_frames(self) -> Iterator[np.ndarray]:
        try:
            while self.audio_buffer_samples < (lookahead_seconds * sample_rate):
                for frame in self._get_resampled_and_gained_next_frames_from_track():
                    self.audio_buffer.append(frame)
                    self.audio_buffer_samples += frame.shape[1]

            while True:
                for frame in self._get_resampled_and_gained_next_frames_from_track():
                    self.audio_buffer.append(frame)
                    self.audio_buffer_samples += frame.shape[1]
                    yield_frame = self.audio_buffer.popleft()
                    self.audio_buffer_samples -= yield_frame.shape[1]
                    yield yield_frame
        except AudioTrackNoMoreFramesError:
            logging.debug(f"Finished decoding {self.path}")
        except Exception as e:
            logging.error(f"Error decoding {self.path}: {e}")
            raise AudioTrackDecodeError(self.path) from e

        self._trim_trailing_silence()

        while self.audio_buffer_samples > (crossfade_seconds * sample_rate):
            yield_frame = self.audio_buffer.popleft()
            self.audio_buffer_samples -= yield_frame.shape[1]
            yield yield_frame

        raise AudioTrackEOFError()


AudioTrackConstructor = type[AudioTrack]
