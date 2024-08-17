from typing import Callable, Literal, Any, Sequence, Concatenate
from typings.extra import (
    P,
    T,
    SongCoverNameUpdateKey,
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


from backend.generate_song_cover import (
    get_named_song_dirs,
    get_song_cover_name,
)
from backend.manage_audio import get_output_audio

PROGRESS_BAR = gr.Progress()


def exception_harness(fun: Callable[P, T]) -> Callable[P, T]:
    def _wrapped_fun(*args: P.args, **kwargs: P.kwargs) -> T:
        try:
            return fun(*args, **kwargs)
        except Exception as e:
            raise gr.Error(str(e))

    return _wrapped_fun


def confirmation_harness(fun: Callable[P, T]) -> Callable[Concatenate[bool, P], T]:
    def _wrapped_fun(confirm: bool, *args: P.args, **kwargs: P.kwargs) -> T:
        if confirm:
            return exception_harness(fun)(*args, **kwargs)
        else:
            raise gr.Error("Confirmation missing!")

    return _wrapped_fun


def confirm_box_js(msg: str) -> str:
    formatted_msg = f"'{msg}'"
    return f"(x) => confirm({formatted_msg})"


def identity(x: T) -> T:
    return x


def update_value(x: Any) -> dict[str, Any]:
    return gr.update(value=x)


def update_dropdowns(
    fn: Callable[P, DropdownChoices],
    num_components: int,
    value: DropdownValue = None,
    value_indices: Sequence[int] = [],
    *args: P.args,
    **kwargs: P.kwargs,
) -> gr.Dropdown | tuple[gr.Dropdown, ...]:
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
    num_components: int,
    value: DropdownValue = None,
    value_indices: Sequence[int] = [],
) -> gr.Dropdown | tuple[gr.Dropdown, ...]:
    return update_dropdowns(get_named_song_dirs, num_components, value, value_indices)


def update_output_audio(
    num_components: int,
    value: DropdownValue = None,
    value_indices: Sequence[int] = [],
) -> gr.Dropdown | tuple[gr.Dropdown, ...]:
    return update_dropdowns(get_output_audio, num_components, value, value_indices)


def toggle_visible_component(
    num_components: int, visible_index: int
) -> dict[str, Any] | tuple[dict[str, Any], ...]:
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
    if num_components == 1:
        return gr.update(interactive=interactive)
    return tuple(gr.update(interactive=interactive) for _ in range(num_components))


def show_hop_slider(
    pitch_detection_algo: F0Method,
) -> gr.Slider:
    if pitch_detection_algo == "mangio-crepe":
        return gr.Slider(visible=True)
    else:
        return gr.Slider(visible=False)


def get_song_cover_name_harness(
    mixed_vocals: str | None = None,
    song_dir: str | None = None,
    voice_model: str | None = None,
    update_key: SongCoverNameUpdateKey = "value",
) -> gr.Textbox:
    update_args: TextBoxArgs = {}
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
    fn: Callable[..., Any]
    inputs: Sequence[Component] | None = None
    outputs: Sequence[Component] | None = None
    name: Literal["click", "success", "then"] = "success"
    show_progress: Literal["full", "minimal", "hidden"] = "full"


def setup_consecutive_event_listeners(
    component: Component, event_args_list: list[EventArgs]
) -> Dependency | Component:
    if len(event_args_list) == 0:
        raise ValueError("Event args list must not be empty.")
    dependency = component
    for event_args in event_args_list:
        event_listener: Callable[..., Dependency] = getattr(dependency, event_args.name)
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
