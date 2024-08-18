from typing import TypeVar, Callable, Any, Literal, ParamSpec, TypedDict, Sequence
from os import PathLike

P = ParamSpec("P")
T = TypeVar("T")

StrOrBytesPath = str | bytes | PathLike[str] | PathLike[bytes]

DropdownChoices = Sequence[str | int | float | tuple[str, str | int | float]] | None

DropdownValue = (
    str | int | float | Sequence[str | int | float] | Callable[..., Any] | None
)

InputType = Literal["yt", "local"]

SongCoverNameUpdateKey = Literal["value", "placeholder"]

F0Method = Literal["rmvpe", "mangio-crepe"]

InputAudioExt = Literal["mp3", "wav", "flac", "aac", "m4a", "ogg"]

OutputAudioExt = Literal["mp3", "wav", "flac", "adts", "ipod", "ogg"]


ModelsTable = list[list[str]]

ModelsTablePredicate = Callable[[dict[str, str | list[str]]], bool]


class ComponentVisibilityKwArgs(TypedDict):
    visible: bool
    value: Any


class UpdateDropdownArgs(TypedDict, total=False):
    choices: DropdownChoices | None
    value: DropdownValue | None


class TextBoxArgs(TypedDict, total=False):
    value: str | None
    placeholder: str | None


class TransferUpdateArgs(TypedDict, total=False):
    value: str | None


RunPipelineHarnessArgs = tuple[
    str,  # song_input
    str,  # voice_model
    int,  # pitch_change_vocals
    int,  # pitch_change_all
    float,  # index_rate
    int,  # filter_radius
    float,  # rms_mix_rate
    float,  # protect
    F0Method,  # f0_method
    int,  # crepe_hop_length
    float,  # reverb_rm_size
    float,  # reverb_wet
    float,  # reverb_dry
    float,  # reverb_damping
    int,  # main_gain
    int,  # inst_gain
    int,  # backup_gain
    int,  # output_sr
    InputAudioExt,  # output_format
    str,  # output_name
    bool,  # return_files
]
