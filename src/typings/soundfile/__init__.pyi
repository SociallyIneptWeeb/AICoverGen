from typing import Literal

from os import PathLike

import numpy as np
from numpy.typing import NDArray

DEFAULT_NDARRAY = NDArray[np.float64 | np.float32 | np.int32 | np.int16]

def read(
    file: int | str | PathLike[str] | PathLike[bytes],
    frames: int = -1,
    start: int = 0,
    stop: int | None = None,
    dtype: Literal["float64", "float32", "int32", "int16"] = "float64",
    always_2d: bool = False,
    fill_value: float | None = None,
    out: DEFAULT_NDARRAY | None = None,
    samplerate: int | None = None,
    channels: int | None = None,
    format: str | None = None,
    subtype: str | None = None,
    endian: Literal["FILE", "LITTLE", "BIG", "CPU"] | None = None,
    closefd: bool | None = True,
) -> tuple[DEFAULT_NDARRAY, int]: ...
def write(
    file: int | str | PathLike[str] | PathLike[bytes],
    data: DEFAULT_NDARRAY,
    samplerate: int,
    subtype: str | None = None,
    endian: Literal["FILE", "LITTLE", "BIG", "CPU"] | None = None,
    format: str | None = None,
    closefd: bool | None = True,
) -> None: ...
