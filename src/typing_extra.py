"""Extra typing for backend and frontend."""

from typing import ParamSpec, TypeVar

from collections.abc import Mapping, Sequence
from enum import IntEnum, StrEnum
from os import PathLike

P = ParamSpec("P")
T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")


StrPath = str | PathLike[str]

Json = Mapping[str, "Json"] | Sequence["Json"] | str | int | float | bool | None


class F0Method(StrEnum):
    """The method to use for pitch detection."""

    RMVPE = "rmvpe"
    MANGIO_CREPE = "mangio-crepe"


class SampleRate(IntEnum):
    """The sample rate of an audio file."""

    HZ_16000 = 16000
    HZ_44100 = 44100
    HZ_48000 = 48000
    HZ_96000 = 96000
    HZ_192000 = 192000


class AudioExt(StrEnum):
    """Audio file formats."""

    MP3 = "mp3"
    WAV = "wav"
    FLAC = "flac"
    OGG = "ogg"
    M4A = "m4a"
    AAC = "aac"
