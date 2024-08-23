"""
Module containing common utility functions and classes for the frontend.
"""

from typing import Callable, Literal, Any, Sequence, Concatenate
from typings.extra import (
    P,
    T,
    F0Method,
    DropdownChoices,
    DropdownValue,
    UpdateDropdownArgs,
    ComponentVisibilityKwArgs,
    TextBoxArgs,
)
from dataclasses import dataclass
from functools import partial

import gradio as gr
from gradio.components.base import Component
from gradio.events import Dependency


from backend.generate_song_cover import get_named_song_dirs, get_song_cover_name
from backend.manage_audio import get_output_audio

PROGRESS_BAR = gr.Progress()


def exception_harness(fun: Callable[P, T]) -> Callable[P, T]:
    """
    Wrap a function in a harness that catches exceptions
    and re-raises them as instances of `gradio.Error`.

    Parameters
    ----------
    fun : Callable[P, T]
        The function to wrap.

    Returns
    -------
    Callable[P, T]
        The wrapped function.
    """

    def _wrapped_fun(*args: P.args, **kwargs: P.kwargs) -> T:
        try:
            return fun(*args, **kwargs)
        except Exception as e:
            raise gr.Error(str(e))

    return _wrapped_fun


def confirmation_harness(fun: Callable[P, T]) -> Callable[Concatenate[bool, P], T]:
    """
    Wrap a function in a harness that requires a confirmation
    before executing and catches exceptions,
    re-raising them as instances of `gradio.Error`.

    Parameters
    ----------
    fun : Callable[P, T]
        The function to wrap.

    Returns
    -------
    Callable[Concatenate[bool, P], T]
        The wrapped function.
    """

    def _wrapped_fun(confirm: bool, *args: P.args, **kwargs: P.kwargs) -> T:
        if confirm:
            return exception_harness(fun)(*args, **kwargs)
        else:
            raise gr.Error("Confirmation missing!")

    return _wrapped_fun


def confirm_box_js(msg: str) -> str:
    """
    Generate JavaScript code for a confirmation box.

    Parameters
    ----------
    msg : str
        Message to display in the confirmation box.

    Returns
    -------
    str
        JavaScript code for the confirmation box.
    """
    formatted_msg = f"'{msg}'"
    return f"(x) => confirm({formatted_msg})"


def identity(x: T) -> T:
    """
    Identity function.

    Parameters
    ----------
    x : T
        Value to return.

    Returns
    -------
    T
        The value.
    """
    return x


def update_value(x: Any) -> dict[str, Any]:
    """
    Update the value of a component.

    Parameters
    ----------
    x : Any
        New value for the component.

    Returns
    -------
    dict[str, Any]
        Dictionary which updates the value of the component.
    """
    return gr.update(value=x)


def update_dropdowns(
    fn: Callable[P, DropdownChoices],
    num_components: int,
    value: DropdownValue = None,
    value_indices: Sequence[int] = [],
    *args: P.args,
    **kwargs: P.kwargs,
) -> gr.Dropdown | tuple[gr.Dropdown, ...]:
    """
    Update the choices and optionally the value of one or more dropdown components.

    Parameters
    ----------
    fn : Callable[P, DropdownChoices]
        Function to get updated choices for the dropdown components.
    num_components : int
        Number of dropdown components to update.
    value : DropdownValue, optional
        New value for dropdown components.
    value_indices : Sequence[int], default=[]
        Indices of dropdown components to update the value for.
    args : P.args
        Positional arguments to pass to the function used to update choices.
    kwargs : P.kwargs
        Keyword arguments to pass to the function used to update choices.

    Returns
    -------
    gr.Dropdown|tuple[gr.Dropdown,...]
        Updated dropdown component or components.

    Raises
    ------
    ValueError
        If value indices are not unique or if an index exceeds the number of components.
    """
    if len(value_indices) != len(set(value_indices)):
        raise ValueError("Value indices must be unique.")
    if value_indices and max(value_indices) >= num_components:
        raise ValueError(
            "Index of a component to update value for exceeds number of components."
        )
    updated_choices = fn(*args, **kwargs)
    update_args: list[UpdateDropdownArgs] = [
        {"choices": updated_choices} for _ in range(num_components)
    ]
    for index in value_indices:
        update_args[index]["value"] = value
    if len(update_args) == 1:
        # NOTE This is a workaround as gradio does not support
        # singleton tuples for components.
        return gr.Dropdown(**update_args[0])
    return tuple(gr.Dropdown(**update_arg) for update_arg in update_args)


def update_cached_input_songs(
    num_components: int, value: DropdownValue = None, value_indices: Sequence[int] = []
) -> gr.Dropdown | tuple[gr.Dropdown, ...]:
    """
    Updates the choices of one or more dropdown components
    to the current set of cached input songs.

    Optionally updates the default value of one or more of these components.

    Parameters
    ----------
    num_components : int
        Number of dropdown components to update.
    value : DropdownValue, optional
        New value for dropdown components.
    value_indices : Sequence[int], default=[]
        Indices of dropdown components to update the value for.

    Returns
    -------
    gr.Dropdown|tuple[gr.Dropdown,...]
        Updated dropdown component or components.
    """
    return update_dropdowns(get_named_song_dirs, num_components, value, value_indices)


def update_output_audio(
    num_components: int, value: DropdownValue = None, value_indices: Sequence[int] = []
) -> gr.Dropdown | tuple[gr.Dropdown, ...]:
    """
    Updates the choices of one or more dropdown
    components to the current set of output audio files.

    Optionally updates the default value of one or more of these components.

    Parameters
    ----------
    num_components : int
        Number of dropdown components to update.
    value : DropdownValue, optional
        New value for dropdown components.
    value_indices : Sequence[int], default=[]
        Indices of dropdown components to update the value for.

    Returns
    -------
    gr.Dropdown|tuple[gr.Dropdown,...]
        Updated dropdown component or components.
    """
    return update_dropdowns(get_output_audio, num_components, value, value_indices)


def toggle_visible_component(
    num_components: int, visible_index: int
) -> dict[str, Any] | tuple[dict[str, Any], ...]:
    """
    Reveal a single component from a set of components.
    All other components are hidden.

    Parameters
    ----------
    num_components : int
        Number of components to set visibility for.
    visible_index : int
        Index of the component to reveal.

    Returns
    -------
    dict|tuple[dict,...]
        A single dictionary or a tuple of dictionaries
        that update the visibility of the components.
    """
    if visible_index >= num_components:
        raise ValueError("Visible index must be less than number of components.")
    update_args: list[ComponentVisibilityKwArgs] = [
        {"visible": False, "value": None} for _ in range(num_components)
    ]
    update_args[visible_index]["visible"] = True
    if num_components == 1:
        return gr.update(**update_args[0])
    return tuple(gr.update(**update_arg) for update_arg in update_args)


def _toggle_component_interactivity(
    num_components: int, interactive: bool
) -> dict[str, Any] | tuple[dict[str, Any], ...]:
    """
    Toggle interactivity of one or more components.

    Parameters
    ----------
    num_components : int
        Number of components to toggle interactivity for.
    interactive : bool
        Whether to make the components interactive or not.

    Returns
    -------
    dict|tuple[dict,...]
        A single dictionary or a tuple of dictionaries
        that update the interactivity of the components.
    """
    if num_components == 1:
        return gr.update(interactive=interactive)
    return tuple(gr.update(interactive=interactive) for _ in range(num_components))


def show_hop_slider(pitch_detection_algo: F0Method) -> gr.Slider:
    """
    Show or hide a slider component based on the given pitch extraction algorithm.

    Parameters
    ----------
    pitch_detection_algo : F0Method
        Pitch detection algorithm to determine visibility of the slider.

    Returns
    -------
    gr.Slider
        Slider component with visibility set accordingly.
    """
    if pitch_detection_algo == "mangio-crepe":
        return gr.Slider(visible=True)
    else:
        return gr.Slider(visible=False)


def update_song_cover_name(
    mixed_vocals: str | None = None,
    song_dir: str | None = None,
    voice_model: str | None = None,
    update_placeholder: bool = False,
) -> gr.Textbox:
    """
    Updates a textbox component so that it displays a suitable name for a cover of
    a given song.

    If the path of an existing song directory is provided, the original song
    name is inferred from that directory. If a voice model is not provided
    but the path of an existing song directory and the path of a mixed vocals file
    in that directory are provided, then the voice model is inferred from
    the mixed vocals file.


    Parameters
    ----------
    mixed_vocals : str, optional
        The path to a mixed vocals file.
    song_dir : str, optional
        The path to a song directory.
    voice_model : str, optional
        The name of a voice model.
    update_placeholder : bool, default=False
        Whether to update the placeholder text of the textbox component.

    Returns
    -------
    gr.Textbox
        Updated textbox component.
    """
    update_args: TextBoxArgs = {}
    update_key = "placeholder" if update_placeholder else "value"
    if mixed_vocals or song_dir or voice_model:
        name = exception_harness(get_song_cover_name)(
            mixed_vocals, song_dir, voice_model, progress_bar=PROGRESS_BAR
        )
        update_args[update_key] = name
    else:
        update_args[update_key] = None
    return gr.Textbox(**update_args)


@dataclass
class EventArgs:
    """
    Data class to store arguments for setting up event listeners.

    Attributes
    ----------
    fn : Callable[..., Any]
        Function to call when an event is triggered.
    inputs : Sequence[Component], optional
        Components to serve as inputs to the function.
    outputs : Sequence[Component], optional
        Components where to store the outputs of the function.
    name : Literal["click", "success", "then"], default="success"
        Name of the event to listen for.
    show_progress : Literal["full", "minimal", "hidden"], default="full"
        Level of progress bar to show when the event is triggered.
    """

    fn: Callable[..., Any]
    inputs: Sequence[Component] | None = None
    outputs: Sequence[Component] | None = None
    name: Literal["click", "success", "then"] = "success"
    show_progress: Literal["full", "minimal", "hidden"] = "full"


def setup_consecutive_event_listeners(
    component: Component, event_args_list: list[EventArgs]
) -> Dependency | Component:
    """
    Set up a chain of event listeners on a component.

    Parameters
    ----------
    component : Component
        The component to set up event listeners on.
    event_args_list : list[EventArgs]
        List of event arguments to set up event listeners with.

    Returns
    -------
    Dependency | Component
        The last dependency in the chain of event listeners.
    """
    if len(event_args_list) == 0:
        raise ValueError("Event args list must not be empty.")
    dependency = component
    for event_args in event_args_list:
        event_listener = getattr(dependency, event_args.name)
        dependency = event_listener(
            event_args.fn,
            inputs=event_args.inputs,
            outputs=event_args.outputs,
            show_progress=event_args.show_progress,
        )
    return dependency


def setup_consecutive_event_listeners_with_toggled_interactivity(
    component: Component,
    event_args_list: list[EventArgs],
    toggled_components: Sequence[Component],
) -> Dependency | Component:
    """
    Set up a chain of event listeners on a component
    with interactivity toggled for a set of other components.

    While the chain of event listeners is being executed,
    the other components are made non-interactive.
    When the chain of event listeners is completed,
    the other components are made interactive again.

    Parameters
    ----------
    component : Component
        The component to set up event listeners on.

    event_args_list : list[EventArgs]
        List of event arguments to set up event listeners with.

    toggled_components : Sequence[Component]
        Components to toggle interactivity for.

    Returns
    -------
    Dependency | Component
        The last dependency in the chain of event listeners.
    """
    if len(event_args_list) == 0:
        raise ValueError("Event args list must not be empty.")

    disable_event_args = EventArgs(
        partial(_toggle_component_interactivity, len(toggled_components), False),
        outputs=toggled_components,
        name="click",
        show_progress="hidden",
    )
    enable_event_args = EventArgs(
        partial(_toggle_component_interactivity, len(toggled_components), True),
        outputs=toggled_components,
        name="then",
        show_progress="hidden",
    )
    event_args_list_augmented = (
        [disable_event_args] + event_args_list + [enable_event_args]
    )
    return setup_consecutive_event_listeners(component, event_args_list_augmented)
