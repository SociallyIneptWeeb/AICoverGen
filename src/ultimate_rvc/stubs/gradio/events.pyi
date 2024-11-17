from typing import Any, Literal, NotRequired, Protocol, Self, TypedDict

import dataclasses
from collections import UserString
from collections.abc import Callable, Sequence
from collections.abc import Set as AbstractSet

from _typeshed import SupportsKeysAndGetItem

from gradio.blocks import Block, BlockContext, Component
from gradio.components import Timer
from gradio.data_classes import FileData, FileDataDict

type Dependency = _Dependency[Any, Any, Any]
type EventListenerCallable = _EventListenerCallable[Any, Any, Any]
type EventListener = _EventListener[Any, Any, Any]

class _EventListenerCallable[T, V, **P](Protocol):
    def __call__(
        self,
        fn: Callable[P, T] | Literal["decorator"] | None = "decorator",
        inputs: (
            Component
            | BlockContext
            | Sequence[Component | BlockContext]
            | AbstractSet[Component | BlockContext]
            | None
        ) = None,
        outputs: (
            Component
            | BlockContext
            | Sequence[Component | BlockContext]
            | AbstractSet[Component | BlockContext]
            | None
        ) = None,
        api_name: str | Literal[False] | None = None,
        scroll_to_output: bool = False,
        show_progress: Literal["full", "minimal", "hidden"] = "full",
        queue: bool = True,
        batch: bool = False,
        max_batch_size: int = 4,
        preprocess: bool = True,
        postprocess: bool = True,
        cancels: Dependency | list[Dependency] | None = None,
        trigger_mode: Literal["once", "multiple", "always_last"] | None = None,
        js: str | None = None,
        concurrency_limit: int | Literal["default"] | None = "default",
        concurrency_id: str | None = None,
        show_api: bool = True,
        stream_every: float = 0.5,
        like_user_message: bool = False,
    ) -> _Dependency[T, V, P]: ...

class _EventListenerCallableFull[T, V, **P](Protocol):
    def __call__(
        self,
        block: Block | None,
        fn: Callable[P, T] | Literal["decorator"] | None = "decorator",
        inputs: (
            Component
            | BlockContext
            | Sequence[Component | BlockContext]
            | AbstractSet[Component | BlockContext]
            | None
        ) = None,
        outputs: (
            Component
            | BlockContext
            | Sequence[Component | BlockContext]
            | AbstractSet[Component | BlockContext]
            | None
        ) = None,
        api_name: str | Literal[False] | None = None,
        scroll_to_output: bool = False,
        show_progress: Literal["full", "minimal", "hidden"] = "full",
        queue: bool = True,
        batch: bool = False,
        max_batch_size: int = 4,
        preprocess: bool = True,
        postprocess: bool = True,
        cancels: Dependency | list[Dependency] | None = None,
        trigger_mode: Literal["once", "multiple", "always_last"] | None = None,
        js: str | None = None,
        concurrency_limit: int | Literal["default"] | None = "default",
        concurrency_id: str | None = None,
        show_api: bool = True,
        time_limit: int | None = None,
        stream_every: float = 0.5,
        like_user_message: bool = False,
    ) -> _Dependency[T, V, P]: ...

def set_cancel_events(
    triggers: Sequence[EventListenerMethod],
    cancels: Dependency | list[Dependency] | None,
) -> None: ...

class _Dependency[T, V, **P](dict[str, V]):
    fn: Callable[P, T]
    associated_timer: Timer | None
    then: EventListenerCallable
    success: EventListenerCallable

    def __init__(
        self,
        trigger: Block | None,
        key_vals: SupportsKeysAndGetItem[str, V],
        dep_index: int | None,
        fn: Callable[P, T],
        associated_timer: Timer | None = None,
    ) -> None: ...
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T: ...

class EventData[T]:
    target: Block | None
    _data: T

    def __init__(self, target: Block | None, _data: T) -> None: ...

class _SelectData(TypedDict):
    index: int | tuple[int, int]
    value: Any
    row_value: NotRequired[list[Any]]
    col_value: NotRequired[list[Any]]
    selected: NotRequired[bool]

class SelectData(EventData[_SelectData]):
    index: int | tuple[int, int]
    value: Any
    row_value: list[Any] | None
    col_value: list[Any] | None
    selected: bool

    def __init__(self, target: Block | None, data: _SelectData) -> None: ...

class _KeyUpData(TypedDict):
    key: str
    input_value: str

class KeyUpData(EventData[_KeyUpData]):
    key: str
    input_value: str

    def __init__(self, target: Block | None, data: _KeyUpData) -> None: ...

class DeletedFileData(EventData[FileDataDict]):
    file: FileData

    def __init__(self, target: Block | None, data: FileDataDict) -> None: ...

class _LikeData(TypedDict):
    index: int | tuple[int, int]
    value: Any
    liked: NotRequired[bool]

class LikeData(EventData[_LikeData]):
    index: int | tuple[int, int]
    value: Any
    liked: bool

    def __init__(self, target: Block | None, data: _LikeData) -> None: ...

class _RetryData(TypedDict):
    index: int | tuple[int, int]
    value: Any

class RetryData(EventData[_RetryData]):
    index: int | tuple[int, int]
    value: Any

    def __init__(self, target: Block | None, data: _RetryData) -> None: ...

class _UndoData(TypedDict):
    index: int | tuple[int, int]
    value: Any

class UndoData(EventData[_UndoData]):
    index: int | tuple[int, int]
    value: Any

    def __init__(self, target: Block | None, data: _UndoData) -> None: ...

class DownloadData(EventData[FileDataDict]):
    file: FileData

    def __init__(self, target: Block | None, data: FileDataDict) -> None: ...

@dataclasses.dataclass
class EventListenerMethod:
    block: Block | None
    event_name: str

class _EventListener[T, V, **P](UserString):
    __slots__ = (
        "callback",
        "config_data",
        "connection",
        "doc",
        "event_name",
        "event_specific_args",
        "has_trigger",
        "listener",
        "show_progress",
        "trigger_after",
        "trigger_only_on_success",
    )

    event_name: str
    has_trigger: bool
    config_data: Callable[..., dict[str, T]]
    show_progress: Literal["full", "minimal", "hidden"]
    callback: Callable[[Block], None] | None
    trigger_after: int | None
    trigger_only_on_success: bool
    doc: str
    connection: Literal["sse", "stream"]
    event_specific_args: list[dict[str, str]]
    listener: _EventListenerCallableFull[T, V, P]

    def __new__(
        cls,
        event_name: str,
        has_trigger: bool = True,
        config_data: Callable[..., dict[str, T]] = dict,  # noqa: PYI011
        show_progress: Literal["full", "minimal", "hidden"] = "full",
        callback: Callable[[Block], None] | None = None,
        trigger_after: int | None = None,
        trigger_only_on_success: bool = False,
        doc: str = "",
        connection: Literal["sse", "stream"] = "sse",
        event_specific_args: list[dict[str, str]] | None = None,
    ) -> Self: ...
    def __init__(
        self,
        event_name: str,
        has_trigger: bool = True,
        config_data: Callable[..., dict[str, T]] = dict,  # noqa: PYI011
        show_progress: Literal["full", "minimal", "hidden"] = "full",
        callback: Callable[[Block], None] | None = None,
        trigger_after: int | None = None,
        trigger_only_on_success: bool = False,
        doc: str = "",
        connection: Literal["sse", "stream"] = "sse",
        event_specific_args: list[dict[str, str]] | None = None,
    ) -> None: ...
    def set_doc(self, component: str) -> None: ...
    def copy(self) -> _EventListener[T, V, P]: ...
    @staticmethod
    def _setup(
        _event_name: str,
        _has_trigger: bool,
        _show_progress: Literal["full", "minimal", "hidden"],
        _callback: Callable[[Block], None] | None,
        _trigger_after: int | None,
        _trigger_only_on_success: bool,
        _event_specific_args: list[dict[str, str]],
        _connection: Literal["sse", "stream"] = "sse",
    ) -> _EventListenerCallableFull[T, V, P]: ...

def on[T, **P](
    triggers: Sequence[EventListenerCallable] | EventListenerCallable | None = None,
    fn: Callable[P, T] | Literal["decorator"] | None = "decorator",
    inputs: (
        Component
        | BlockContext
        | Sequence[Component | BlockContext]
        | AbstractSet[Component | BlockContext]
        | None
    ) = None,
    outputs: (
        Component
        | BlockContext
        | Sequence[Component | BlockContext]
        | AbstractSet[Component | BlockContext]
        | None
    ) = None,
    *,
    api_name: str | Literal[False] | None = None,
    scroll_to_output: bool = False,
    show_progress: Literal["full", "minimal", "hidden"] = "full",
    queue: bool = True,
    batch: bool = False,
    max_batch_size: int = 4,
    preprocess: bool = True,
    postprocess: bool = True,
    cancels: Dependency | list[Dependency] | None = None,
    trigger_mode: Literal["once", "multiple", "always_last"] | None = None,
    js: str | None = None,
    concurrency_limit: int | Literal["default"] | None = "default",
    concurrency_id: str | None = None,
    show_api: bool = True,
    time_limit: int | None = None,
    stream_every: float = 0.5,
) -> _Dependency[T, Any, P]: ...

class Events:
    change: EventListener
    input: EventListener
    click: EventListener
    double_click: EventListener
    submit: EventListener
    edit: EventListener
    clear: EventListener
    play: EventListener
    pause: EventListener
    stop: EventListener
    end: EventListener
    start_recording: EventListener
    pause_recording: EventListener
    stop_recording: EventListener
    focus: EventListener
    blur: EventListener
    upload: EventListener
    release: EventListener
    select: EventListener
    stream: EventListener
    like: EventListener
    example_select: EventListener
    load: EventListener
    key_up: EventListener
    apply: EventListener
    delete: EventListener
    tick: EventListener
    undo: EventListener
    retry: EventListener
    expand: EventListener
    collapse: EventListener
    download: EventListener

__all__ = [
    "DeletedFileData",
    "Dependency",
    "DownloadData",
    "EventData",
    "EventListener",
    "EventListenerMethod",
    "Events",
    "KeyUpData",
    "LikeData",
    "RetryData",
    "SelectData",
    "UndoData",
    "on",
    "set_cancel_events",
]
