"""Utility functions for voice conversion."""

import numpy as np
from numpy.typing import NDArray

import ffmpeg


def load_audio(file: str, sr: int) -> NDArray[np.float32]:
    """
    Load an audio file into a numpy array with a target sample rate.

    A subprocess is launched to decode the given audio file while
    down-mixing and resampling as necessary.

    Parameters
    ----------
    file : str
        Path to the audio file.
    sr : int
        Target sample rate.

    Returns
    -------
    NDArray[np.float32]
        Decoded audio file in numpy array format.

    Raises
    ------
    RuntimeError
        If the audio file cannot be loaded.

    See Also
    --------
    https://github.com/openai/whisper/blob/main/whisper/audio.py#L26

    Notes
    -----
    Requires the ffmpeg CLI and `typed-ffmpeg` package to be installed.

    """
    try:
        # NOTE prevent the input path from containing spaces and
        # carriage returns at the beginning and end.
        file = file.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        out, _ = (
            ffmpeg.input(file, threads=0)
            .output(
                filename="-",
                f="f32le",
                acodec="pcm_f32le",
                ac=1,
                ar=sr,
            )
            .run(
                cmd=["ffmpeg", "-nostdin"],
                capture_stdout=True,
                capture_stderr=True,
            )
        )

    except Exception as e:
        err_msg = f"Failed to load audio: {e}"
        raise RuntimeError(err_msg) from e

    return np.frombuffer(out, np.float32).flatten()
