#########################################################################
# Local stubs for python-soxr.
#########################################################################

from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import ArrayLike

QQ: int
LQ: int
MQ: int
HQ: int
VHQ: int

Quality: TypeAlias = Literal["QQ", "LQ", "MQ", "HQ", "VHQ"] | int
SoxrDType: TypeAlias = (
    Literal["float32", "float64", "int16", "int32"]
    | type[np.float32]
    | type[np.float64]
    | type[np.int16]
    | type[np.int32]
)

def resample(
    x: ArrayLike,
    in_rate: float,
    out_rate: float,
    quality: Quality = "HQ",
) -> np.ndarray: ...

class ResampleStream:
    def __init__(
        self,
        in_rate: float,
        out_rate: float,
        num_channels: int,
        dtype: SoxrDType = "float32",
        quality: Quality = "HQ",
    ) -> None: ...
    def clear(self) -> None: ...
    def delay(self) -> float: ...
    def num_clips(self) -> int: ...
    def resample_chunk(self, x: np.ndarray, last: bool = False) -> np.ndarray: ...
