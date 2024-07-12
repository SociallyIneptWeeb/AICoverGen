from typing import TypeVar, Callable, Any, Literal
from os import PathLike
from typing_extensions import ParamSpec, TypedDict

P = ParamSpec("P")
T = TypeVar("T")

StrOrBytesPath = str | bytes | PathLike[str] | PathLike[bytes]

InputChoices = list[tuple[str, str]] | list[str]

DropdownChoices = list[str | int | float | tuple[str, str | int | float]] | None

DropdownValue = str | int | float | list[str | int | float] | Callable[..., Any] | None

InputType = Literal["yt", "local"]

SongCoverNameUpdateKey = Literal["value", "placeholder"]

F0Method = Literal["rmvpe", "mangio-crepe"]

InputAudioExt = Literal[
    "mp3",
    "wav",
    "flac",
    "aac",
    "m4a",
    "ogg",
]

OutputAudioExt = Literal[
    "mp3",
    "wav",
    "flac",
    "adts",
    "ipod",
    "ogg",
]


ModelsTable = list[list[str]]

ModelsTablePredicate = Callable[[dict[str, str]], bool]


class ComponentVisibilityUpdate(TypedDict):
    __type__: str
    visible: bool
    value: None


class ComponentInteractivityUpdate(TypedDict):
    __type__: str
    interactive: bool


class UpdateDropdownArgs(TypedDict, total=False):
    choices: DropdownChoices | None
    value: DropdownValue | None


class TextBoxArgs(TypedDict, total=False):
    value: str | None
    placeholder: str | None


class TransferUpdateArgs(TypedDict, total=False):
    value: str | None


MixSongCoverHarnessArgs = tuple[str, int, int, int, int, InputAudioExt, str, bool]

RunPipelineHarnessArgs = tuple[
    str,
    str,
    int,
    int,
    float,
    int,
    float,
    float,
    F0Method,
    int,
    float,
    float,
    float,
    float,
    int,
    int,
    int,
    int,
    InputAudioExt,
    str,
    bool,
    bool,
]
