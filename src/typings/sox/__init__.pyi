from typing import Self, Any
from numpy.typing import NDArray

class Transformer:
    def pitch(self, n_semitones: float, quick: bool = False) -> Self: ...
    def build_array(
        self,
        input_filepath: str | None = None,
        input_array: NDArray[...] | None = None,
        sample_rate_in: int | None = None,
        extra_args: list[Any] | None = None,
    ) -> NDArray[...]: ...
