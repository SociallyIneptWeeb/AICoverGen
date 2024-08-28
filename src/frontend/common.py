"""
Module defining common utility functions and classes for the
frontend.
"""

from typing import Any, Concatenate, Literal

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial

import gradio as gr
from gradio.components.base import Component
from gradio.events import Dependency

from exceptions import NotProvidedError

from typing_extra import F0Method, P, T

from backend.generate_song_cover import get_named_song_dirs, get_song_cover_name
from backend.manage_audio import get_saved_output_audio

from frontend.typing_extra import (
    ComponentVisibilityKwArgs,
    DropdownChoices,
    DropdownValue,
    TextBoxKwArgs,
    UpdateDropdownKwArgs,
)

PROGRESS_BAR = gr.Progress()


def exception_harness(fn: Callable[P, T]) -> Callable[P, T]:
    """
    Wrap a function in a harness that catches exceptions and re-raises
    them as instances of `gradio.Error`.

    Parameters
    ----------
    fn : Callable[P, T]
        The function to wrap.

    Returns
    -------
    Callable[P, T]
        The wrapped function.

    """

    def _wrapped_fn(*args: P.args, **kwargs: P.kwargs) -> T:
        try:
            return fn(*args, **kwargs)
        except gr.Error:
            raise
        except NotProvidedError as e:
            msg = e.ui_msg or e
            raise gr.Error(str(msg)) from None
        except Exception as e:
            raise gr.Error(str(e)) from e

    return _wrapped_fn


def confirmation_harness(fn: Callable[P, T]) -> Callable[Concatenate[bool, P], T]:
    """
    Wrap a function in a harness that requires a confirmation before
    executing and catches exceptions, re-raising them as instances of
    `gradio.Error`.

    Parameters
    ----------
    fn : Callable[P, T]
        The function to wrap.

    Returns
    -------
    Callable[Concatenate[bool, P], T]
        The wrapped function.

    """

    def _wrapped_fn(confirm: bool, *args: P.args, **kwargs: P.kwargs) -> T:
        if confirm:
            return exception_harness(fn)(*args, **kwargs)
        err_msg = "Confirmation missing!"
        raise gr.Error(err_msg)

    return _wrapped_fn


def render_msg(
    template: str,
    *args: str,
    display_info: bool = False,
    **kwargs: str,
) -> str:
    """
    Render a message template with the provided arguments.

    Parameters
    ----------
    template : str
        Message template to render.
    args : str
        Positional arguments to pass to the template.
    display_info : bool, default=False
        Whether to display the rendered message as an info message
        in addition to returning it.
    kwargs : str
        Keyword arguments to pass to the template.

    Returns
    -------
    str
        Rendered message.

    """
    msg = template.format(*args, **kwargs)
    if display_info:
        gr.Info(msg)
    return msg


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


def update_value(x: str) -> dict[str, Any]:
    """
    Update the value of a component.

    Parameters
    ----------
    x : str
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
    Update the choices and optionally the value of one or more dropdown
    components.

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
        Positional arguments to pass to the function used to update
        dropdown choices.
    kwargs : P.kwargs
        Keyword arguments to pass to the function used to update
        dropdown choices.

    Returns
    -------
    gr.Dropdown | tuple[gr.Dropdown,...]
        Updated dropdown component or components.

    Raises
    ------
    ValueError
        If not all provided indices are unique or if an index exceeds
        or is equal to the number of dropdown components.

    """
    if len(value_indices) != len(set(value_indices)):
        err_msg = "Value indices must be unique."
        raise ValueError(err_msg)
    if value_indices and max(value_indices) >= num_components:
        err_msg = (
            "Index of a dropdown component to update the value for exceeds the number"
            " of dropdown components to update."
        )
        raise ValueError(err_msg)
    updated_choices = fn(*args, **kwargs)
    update_args_list: list[UpdateDropdownKwArgs] = [
        {"choices": updated_choices} for _ in range(num_components)
    ]
    for index in value_indices:
        update_args_list[index]["value"] = value

    match update_args_list:
        case [update_args]:
            # NOTE This is a workaround as gradio does not support
            # singleton tuples for components.
            return gr.Dropdown(**update_args)
        case _:
            return tuple(gr.Dropdown(**update_args) for update_args in update_args_list)


def update_cached_songs(
    num_components: int,
    value: DropdownValue = None,
    value_indices: Sequence[int] = [],
) -> gr.Dropdown | tuple[gr.Dropdown, ...]:
    """
    Update the choices of one or more dropdown components to the set of
    currently cached songs.

    Optionally update the default value of one or more of these
    components.

    Parameters
    ----------
    num_components : int
        Number of dropdown components to update.
    value : DropdownValue, optional
        New value for the dropdown components.
    value_indices : Sequence[int], default=[]
        Indices of dropdown components to update the value for.

    Returns
    -------
    gr.Dropdown | tuple[gr.Dropdown,...]
        Updated dropdown component or components.

    """
    return update_dropdowns(get_named_song_dirs, num_components, value, value_indices)


def update_output_audio(
    num_components: int,
    value: DropdownValue = None,
    value_indices: Sequence[int] = [],
) -> gr.Dropdown | tuple[gr.Dropdown, ...]:
    """
    Update the choices of one or more dropdown components to the set of
    currently saved output audio files.

    Optionally update the default value of one or more of these
    components.

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
    gr.Dropdown | tuple[gr.Dropdown,...]
        Updated dropdown component or components.

    """
    return update_dropdowns(
        get_saved_output_audio,
        num_components,
        value,
        value_indices,
    )


def toggle_visible_component(
    num_components: int,
    visible_index: int,
) -> dict[str, Any] | tuple[dict[str, Any], ...]:
    """
    Reveal a single component from a set of components. All other
    components are hidden.

    Parameters
    ----------
    num_components : int
        Number of components to set visibility for.
    visible_index : int
        Index of the component to reveal.

    Returns
    -------
    dict[str, Any] | tuple[dict[str, Any], ...]
        A single dictionary or a tuple of dictionaries that update the
        visibility of the components.

    Raises
    ------
    ValueError
        If the visible index exceeds or is equal to the number of
        components to set visibility for.

    """
    if visible_index >= num_components:
        err_msg = (
            "Visible index must be less than the number of components to set visibility"
            " for."
        )
        raise ValueError(err_msg)
    update_args_list: list[ComponentVisibilityKwArgs] = [
        {"visible": False, "value": None} for _ in range(num_components)
    ]
    update_args_list[visible_index]["visible"] = True
    match update_args_list:
        case [update_args]:
            return gr.update(**update_args)
        case _:
            return tuple(gr.update(**update_args) for update_args in update_args_list)


def _toggle_component_interactivity(
    num_components: int,
    interactive: bool,
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
    dict[str, Any] | tuple[dict[Str, Any],...]
        A single dictionary or a tuple of dictionaries that update the
        interactivity of the components.

    """
    if num_components == 1:
        return gr.update(interactive=interactive)
    return tuple(gr.update(interactive=interactive) for _ in range(num_components))


def show_hop_slider(f0_method: F0Method) -> gr.Slider:
    """
    Show or hide a slider component based on the provided pitch
    detection method.

    Parameters
    ----------
    f0_method : F0Method
        Pitch detection algorithm to determine visibility of the
        slider.

    Returns
    -------
    gr.Slider
        Slider component with visibility set accordingly.

    """
    return gr.Slider(visible=f0_method == F0Method.MANGIO_CREPE)


def update_song_cover_name(
    effected_vocals_track: str | None = None,
    song_dir: str | None = None,
    model_name: str | None = None,
    update_placeholder: bool = False,
) -> gr.Textbox:
    """
    Update a textbox component so that it displays a suitable name for a
    cover of a given song.

    If the path of an existing song directory is provided, the name of
    the song is inferred from that directory. If the name of a voice
    model is not provided but the path of an existing song directory
    and the path of an effected vocals track in that directory are
    provided, then the voice model is inferred from the effected vocals
    track.


    Parameters
    ----------
    effected_vocals_track : str, optional
        The path to an effected vocals track.
    song_dir : str, optional
        The path to a song directory.
    model_name : str, optional
        The name of a voice model.
    update_placeholder : bool, default=False
        Whether to update the placeholder text instead of the value of
        the textbox component.

    Returns
    -------
    gr.Textbox
        Textbox component with updated value or placeholder text.

    """
    update_args: TextBoxKwArgs = {}
    update_key = "placeholder" if update_placeholder else "value"
    if effected_vocals_track or song_dir or model_name:
        harness_fn = exception_harness(get_song_cover_name)
        song_cover_name = harness_fn(
            effected_vocals_track,
            song_dir,
            model_name,
            progress_bar=PROGRESS_BAR,
        )
        update_args[update_key] = song_cover_name
    else:
        update_args[update_key] = None
    return gr.Textbox(**update_args)


@dataclass
class EventArgs:
    """
    Arguments for setting up an event listener.

    Attributes
    ----------
    fn : Callable[..., Any]
        Function to call when an event is triggered.
    inputs : Sequence[Component], optional
        Components to serve as inputs to the function.
    outputs : Sequence[Component], optional
        Components where outputs of the function are stored.
    name : Literal["click", "success", "then"], default="success"
        Name of the type of event to listen for.
    show_progress : Literal["full", "minimal", "hidden"], default="full"
        Level of the progress bar animation to show when the event is
        triggered.

    """

    fn: Callable[..., Any]
    inputs: Sequence[Component] | None = None
    outputs: Sequence[Component] | None = None
    name: Literal["click", "success", "then"] = "success"
    show_progress: Literal["full", "minimal", "hidden"] = "full"


def _chain_event_listeners(
    component: Component,
    event_args_seq: Sequence[EventArgs],
) -> Dependency | Component:
    """
    Set up a chain of event listeners on a component.

    Parameters
    ----------
    component : Component
        The component to set up a chain of event listeners on.
    event_args_seq : Sequence[EventArgs]
        Sequence of event arguments to set up event listeners with.

    Returns
    -------
    Dependency | Component
        The last dependency in the chain of event listeners.

    Raises
    ------
    ValueError
        If no event arguments are provided.

    """
    if len(event_args_seq) == 0:
        err_msg = (
            "The sequence of event arguments is empty. At least one set of event"
            " arguments must be provided."
        )
        raise ValueError(err_msg)
    dependency: Component | Dependency = component
    for event_args in event_args_seq:
        event_listener = getattr(dependency, event_args.name)
        dependency = event_listener(
            event_args.fn,
            inputs=event_args.inputs,
            outputs=event_args.outputs,
            show_progress=event_args.show_progress,
        )
    return dependency


def chain_event_listeners(
    component: Component,
    event_args_seq: Sequence[EventArgs],
    toggled_components: Sequence[Component],
) -> Dependency | Component:
    """
    Set up a chain of event listeners on a component with interactivity
    toggled for a set of other components.

    While the chain of event listeners is executing, the other
    components are made non-interactive. When the chain of event
    listeners has finished executing, the other components are made
    interactive again.

    Parameters
    ----------
    component : Component
        The component to set up event listeners on.

    event_args_seq : Sequence[EventArgs]
        Sequence of event arguments to set up event listeners with.

    toggled_components : Sequence[Component]
        Components to toggle interactivity for.

    Returns
    -------
    Dependency | Component
        The last dependency in the chain of event listeners.

    Raises
    ------
    ValueError
        If no event arguments are provided.

    """
    if len(event_args_seq) == 0:
        err_msg = (
            "The sequence of event arguments is empty. At least one set of event"
            " arguments must be provided."
        )
        raise ValueError(err_msg)

    disable_event_args = EventArgs(
        partial(
            _toggle_component_interactivity,
            len(toggled_components),
            interactive=False,
        ),
        outputs=toggled_components,
        name="click",
        show_progress="hidden",
    )
    enable_event_args = EventArgs(
        partial(
            _toggle_component_interactivity,
            len(toggled_components),
            interactive=True,
        ),
        outputs=toggled_components,
        name="then",
        show_progress="hidden",
    )
    event_args_seq_augmented = [
        disable_event_args,
        *event_args_seq,
        enable_event_args,
    ]
    return _chain_event_listeners(component, event_args_seq_augmented)
