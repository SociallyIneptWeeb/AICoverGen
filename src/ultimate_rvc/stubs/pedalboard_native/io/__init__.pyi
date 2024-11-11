from typing import Literal, Self, overload

import numpy as np
from numpy.typing import NDArray

class AudioFile:
    @classmethod
    @overload
    def __new__(
        cls: object,
        filename: str,
        mode: Literal["r"] = "r",
    ) -> ReadableAudioFile: ...
    @classmethod
    @overload
    def __new__(
        cls: object,
        filename: str,
        mode: Literal["w"],
        samplerate: float | None = None,
        num_channels: int = 1,
        bit_depth: int = 16,
        quality: str | float | None = None,
    ) -> WriteableAudioFile: ...

class ReadableAudioFile(AudioFile):
    def __enter__(self) -> Self: ...
    def __exit__(self, arg0: object, arg1: object, arg2: object) -> None: ...
    def read(self, num_frames: float = 0) -> NDArray[np.float32]: ...
    def tell(self) -> int: ...
    @property
    def frames(self) -> int: ...
    @property
    def num_channels(self) -> int: ...
    @property
    def samplerate(self) -> float | int: ...

class WriteableAudioFile(AudioFile):
    def __enter__(self) -> Self: ...
    def __exit__(self, arg0: object, arg1: object, arg2: object) -> None: ...
    def write(self, samples: NDArray[...]) -> None: ...
