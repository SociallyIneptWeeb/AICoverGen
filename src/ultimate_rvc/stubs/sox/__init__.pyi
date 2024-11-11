from typing import Self

from pathlib import Path

from numpy.typing import NDArray

class Transformer:
    def pitch(
        self,
        n_semitones: float,
        quick: bool = False,
    ) -> Self: ...
    def build_array(
        self,
        input_filepath: str | Path | None = None,
        input_array: NDArray[...] | None = None,
        sample_rate_in: float | None = None,
        extra_args: list[str] | None = None,
    ) -> NDArray[...]: ...
