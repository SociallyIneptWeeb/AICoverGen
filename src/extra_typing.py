from typing import TypeVar, Optional, Union, Callable, Any, Literal
from os import PathLike
from typing_extensions import ParamSpec, TypedDict

P = ParamSpec("P")
T = TypeVar("T")

StrOrBytesPath = Union[str, bytes, PathLike[str], PathLike[bytes]]

InputChoices = Union[list[tuple[str, str]], list[str]]

DropdownChoices = Optional[
    list[Union[str, int, float, tuple[str, Union[str, int, float]]]]
]

DropdownValue = Union[
    str, int, float, list[Union[str, int, float]], Callable[..., Any], None
]

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
    choices: Optional[DropdownChoices]
    value: Optional[DropdownValue]


class TextBoxArgs(TypedDict, total=False):
    value: Optional[str]
    placeholder: Optional[str]


class TransferUpdateArgs(TypedDict, total=False):
    value: Optional[str]


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
