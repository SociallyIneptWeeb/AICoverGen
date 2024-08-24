from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    AbstractSet,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Self,
    Sequence,
    Union,
)

import dataclasses

from gradio.data_classes import FileData, FileDataDict

if TYPE_CHECKING:
    from gradio.blocks import Block, BlockContext, Component
    from gradio.components import Timer

def set_cancel_events(
    triggers: Sequence[EventListenerMethod],
    cancels: None | dict[str, Any] | list[dict[str, Any]],
) -> None: ...

class Dependency(dict):
    fn: Callable[..., Any]
    associated_timer: Timer | None
    then: Callable[..., Any]
    success: Callable[..., Any]
    def __init__(
        self, trigger, key_vals, dep_index, fn, associated_timer: Timer | None = ...
    ) -> None:
        """
        The Dependency object is usualy not created directly but is returned when an event listener is set up. It contains the configuration
        data for the event listener, and can be used to set up additional event listeners that depend on the completion of the current event
        listener using .then() and .success().

        Demos: chatbot_consecutive, blocks_chained_events
        """
        ...

    def __call__(self, *args, **kwargs) -> Any: ...

class EventData:
    """
    When gr.EventData or one of its subclasses is added as a type hint to an argument of a prediction function, a gr.EventData object will automatically be passed as the value of that argument.
    The attributes of this object contains information about the event that triggered the listener. The gr.EventData object itself contains a `.target` attribute that refers to the component
    that triggered the event, while subclasses of gr.EventData contains additional attributes that are different for each class.

    Example:
        import gradio as gr
        with gr.Blocks() as demo:
            table = gr.Dataframe([[1, 2, 3], [4, 5, 6]])
            gallery = gr.Gallery([("cat.jpg", "Cat"), ("dog.jpg", "Dog")])
            textbox = gr.Textbox("Hello World!")
            statement = gr.Textbox()
            def on_select(value, evt: gr.EventData):
                return f"The {evt.target} component was selected, and its value was {value}."
            table.select(on_select, table, statement)
            gallery.select(on_select, gallery, statement)
            textbox.select(on_select, textbox, statement)
        demo.launch()
    Demos: gallery_selections, tictactoe
    """

    target: Block | None
    _data: Any
    def __init__(self, target: Block | None, _data: Any) -> None:
        """
        Parameters:
            target: The component object that triggered the event. Can be used to distinguish multiple components bound to the same listener.
        """
        ...

class SelectData(EventData):
    """
    The gr.SelectData class is a subclass of gr.EventData that specifically carries information about the `.select()` event. When gr.SelectData
    is added as a type hint to an argument of an event listener method, a gr.SelectData object will automatically be passed as the value of that argument.
    The attributes of this object contains information about the event that triggered the listener.

    Example:
        import gradio as gr
        with gr.Blocks() as demo:
            table = gr.Dataframe([[1, 2, 3], [4, 5, 6]])
            gallery = gr.Gallery([("cat.jpg", "Cat"), ("dog.jpg", "Dog")])
            textbox = gr.Textbox("Hello World!")
            statement = gr.Textbox()
            def on_select(evt: gr.SelectData):
                return f"You selected {evt.value} at {evt.index} from {evt.target}"
            table.select(on_select, table, statement)
            gallery.select(on_select, gallery, statement)
            textbox.select(on_select, textbox, statement)
        demo.launch()
    Demos: gallery_selections, tictactoe
    """

    index: int | tuple[int, int]
    value: Any
    row_value: list[Any] | None
    col_value: list[Any] | None
    selected: bool

    def __init__(self, target: Block | None, data: Any) -> None: ...

class KeyUpData(EventData):
    """
    The gr.KeyUpData class is a subclass of gr.EventData that specifically carries information about the `.key_up()` event. When gr.KeyUpData
    is added as a type hint to an argument of an event listener method, a gr.KeyUpData object will automatically be passed as the value of that argument.
    The attributes of this object contains information about the event that triggered the listener.

    Example:
        import gradio as gr
        def test(value, key_up_data: gr.KeyUpData):
            return {
                "component value": value,
                "input value": key_up_data.input_value,
                "key": key_up_data.key
            }
        with gr.Blocks() as demo:
            d = gr.Dropdown(["abc", "def"], allow_custom_value=True)
            t = gr.JSON()
            d.key_up(test, d, t)
        demo.launch()
    Demos: dropdown_key_up
    """

    key: str
    input_value: str
    def __init__(self, target: Block | None, data: Any) -> None: ...

class DeletedFileData(EventData):
    """
    The gr.DeletedFileData class is a subclass of gr.EventData that specifically carries information about the `.delete()` event. When gr.DeletedFileData
    is added as a type hint to an argument of an event listener method, a gr.DeletedFileData object will automatically be passed as the value of that argument.
    The attributes of this object contains information about the event that triggered the listener.
    Example:
        import gradio as gr
        def test(delete_data: gr.DeletedFileData):
            return delete_data.file.path
        with gr.Blocks() as demo:
            files = gr.File(file_count="multiple")
            deleted_file = gr.File()
            files.delete(test, None, deleted_file)
        demo.launch()
    Demos: file_component_events
    """

    file: FileData
    def __init__(self, target: Block | None, data: FileDataDict) -> None: ...

class LikeData(EventData):
    """
    The gr.LikeData class is a subclass of gr.EventData that specifically carries information about the `.like()` event. When gr.LikeData
    is added as a type hint to an argument of an event listener method, a gr.LikeData object will automatically be passed as the value of that argument.
    The attributes of this object contains information about the event that triggered the listener.
    Example:
        import gradio as gr
        def test(value, like_data: gr.LikeData):
            return {
                "chatbot_value": value,
                "liked_message": like_data.value,
                "liked_index": like_data.index,
                "liked_or_disliked_as_bool": like_data.liked
            }
        with gr.Blocks() as demo:
            c = gr.Chatbot([("abc", "def")])
            t = gr.JSON()
            c.like(test, c, t)
        demo.launch()
    Demos: chatbot_core_components_simple
    """

    def __init__(self, target: Block | None, data: Any) -> None:
        index: int | tuple[int, int]
        value: Any
        liked: bool

@dataclasses.dataclass
class EventListenerMethod:
    block: Block | None
    event_name: str

if TYPE_CHECKING:
    EventListenerCallable = Callable[
        [
            Union[Callable, None],
            Union[Component, Sequence[Component], None],
            Union[Block, Sequence[Block], Sequence[Component], Component, None],
            Union[str, None, Literal[False]],
            bool,
            Literal["full", "minimal", "hidden"],
            Union[bool, None],
            bool,
            int,
            bool,
            bool,
            Union[Dict[str, Any], List[Dict[str, Any]], None],
            Union[float, None],
            Union[Literal["once", "multiple", "always_last"], None],
            Union[str, None],
            Union[int, None, Literal["default"]],
            Union[str, None],
            bool,
        ],
        Dependency,
    ]

class EventListener(str):
    has_trigger: bool
    config_data: Callable[..., dict[str, Any]]
    event_name: str
    show_progress: Literal["full", "minimal", "hidden"]
    trigger_after: int | None
    trigger_only_on_success: bool
    callback: Callable | None
    doc: str
    listener: Callable[..., Dependency]

    def __new__(cls, event_name, *_args, **_kwargs) -> Self: ...
    def __init__(
        self,
        event_name: str,
        has_trigger: bool = ...,
        config_data: Callable[..., dict[str, Any]] = ...,
        show_progress: Literal["full", "minimal", "hidden"] = ...,
        callback: Callable | None = ...,
        trigger_after: int | None = ...,
        trigger_only_on_success: bool = ...,
        doc: str = ...,
    ) -> None: ...
    def set_doc(self, component: str) -> None: ...
    def copy(self) -> EventListener: ...
    @staticmethod
    def _setup(
        _event_name: str,
        _has_trigger: bool,
        _show_progress: Literal["full", "minimal", "hidden"],
        _callback: Callable[..., Any] | None,
        _trigger_after: int | None,
        _trigger_only_on_success: bool,
    ) -> Callable[..., Dependency]: ...

def on(
    triggers: Sequence[EventListenerCallable] | EventListenerCallable | None = ...,
    fn: Callable | None | Literal["decorator"] = ...,
    inputs: (
        Component
        | BlockContext
        | Sequence[Component | BlockContext]
        | AbstractSet[Component | BlockContext]
        | None
    ) = ...,
    outputs: (
        Component
        | BlockContext
        | Sequence[Component | BlockContext]
        | AbstractSet[Component | BlockContext]
        | None
    ) = ...,
    *,
    api_name: str | None | Literal[False] = ...,
    scroll_to_output: bool = ...,
    show_progress: Literal["full", "minimal", "hidden"] = ...,
    queue: bool = ...,
    batch: bool = ...,
    max_batch_size: int = ...,
    preprocess: bool = ...,
    postprocess: bool = ...,
    cancels: dict[str, Any] | list[dict[str, Any]] | None = ...,
    trigger_mode: Literal["once", "multiple", "always_last"] | None = ...,
    every: float | None = ...,
    js: str | None = ...,
    concurrency_limit: int | None | Literal["default"] = ...,
    concurrency_id: str | None = ...,
    show_api: bool = ...,
) -> Dependency:
    """
    Sets up an event listener that triggers a function when the specified event(s) occur. This is especially
    useful when the same function should be triggered by multiple events. Only a single API endpoint is generated
    for all events in the triggers list.

    Parameters:
        triggers: List of triggers to listen to, e.g. [btn.click, number.change]. If None, will listen to changes to any inputs.
        fn: the function to call when this event is triggered. Often a machine learning model's prediction function. Each parameter of the function corresponds to one input component, and the function should return a single value or a tuple of values, with each element in the tuple corresponding to one output component.
        inputs: List of gradio.components to use as inputs. If the function takes no inputs, this should be an empty list.
        outputs: List of gradio.components to use as outputs. If the function returns no outputs, this should be an empty list.
        api_name: Defines how the endpoint appears in the API docs. Can be a string, None, or False. If False, the endpoint will not be exposed in the api docs. If set to None, the endpoint will be exposed in the api docs as an unnamed endpoint, although this behavior will be changed in Gradio 4.0. If set to a string, the endpoint will be exposed in the api docs with the given name.
        scroll_to_output: If True, will scroll to output component on completion
        show_progress: how to show the progress animation while event is running: "full" shows a spinner which covers the output component area as well as a runtime display in the upper right corner, "minimal" only shows the runtime display, "hidden" shows no progress animation at all
        queue: If True, will place the request on the queue, if the queue has been enabled. If False, will not put this event on the queue, even if the queue has been enabled. If None, will use the queue setting of the gradio app.
        batch: If True, then the function should process a batch of inputs, meaning that it should accept a list of input values for each parameter. The lists should be of equal length (and be up to length `max_batch_size`). The function is then *required* to return a tuple of lists (even if there is only 1 output component), with each list in the tuple corresponding to one output component.
        max_batch_size: Maximum number of inputs to batch together if this is called from the queue (only relevant if batch=True)
        preprocess: If False, will not run preprocessing of component data before running 'fn' (e.g. leaving it as a base64 string if this method is called with the `Image` component).
        postprocess: If False, will not run postprocessing of component data before returning 'fn' output to the browser.
        cancels: A list of other events to cancel when this listener is triggered. For example, setting cancels=[click_event] will cancel the click_event, where click_event is the return value of another components .click method. Functions that have not yet run (or generators that are iterating) will be cancelled, but functions that are currently running will be allowed to finish.
        trigger_mode: If "once" (default for all events except `.change()`) would not allow any submissions while an event is pending. If set to "multiple", unlimited submissions are allowed while pending, and "always_last" (default for `.change()` and `.key_up()` events) would allow a second submission after the pending event is complete.
        every: Will be deprecated in favor of gr.Timer. Run this event 'every' number of seconds while the client connection is open. Interpreted in seconds.
        js: Optional frontend js method to run before running 'fn'. Input arguments for js method are values of 'inputs', return should be a list of values for output components.
        concurrency_limit: If set, this is the maximum number of this event that can be running simultaneously. Can be set to None to mean no concurrency_limit (any number of this event can be running simultaneously). Set to "default" to use the default concurrency limit (defined by the `default_concurrency_limit` parameter in `Blocks.queue()`, which itself is 1 by default).
        concurrency_id: If set, this is the id of the concurrency group. Events with the same concurrency_id will be limited by the lowest set concurrency_limit.
        show_api: whether to show this event in the "view API" page of the Gradio app, or in the ".view_api()" method of the Gradio clients. Unlike setting api_name to False, setting show_api to False will still allow downstream apps as well as the Clients to use this event. If fn is None, show_api will automatically be set to False.
    Example:
        import gradio as gr
        with gr.Blocks() as demo:
            with gr.Row():
                input = gr.Textbox()
                button = gr.Button("Submit")
            output = gr.Textbox()
            gr.on(
                triggers=[button.click, input.submit],
                fn=lambda x: x,
                inputs=[input],
                outputs=[output]
            )
        demo.launch()
    """
    ...

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
    load: EventListener
    key_up: EventListener
    apply: EventListener
    delete: EventListener
    tick: EventListener

__all__ = [
    "set_cancel_events",
    "Dependency",
    "EventData",
    "SelectData",
    "KeyUpData",
    "DeletedFileData",
    "LikeData",
    "EventListenerMethod",
    "EventListener",
    "on",
    "Events",
]
