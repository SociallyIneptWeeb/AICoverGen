from typing import Callable, Literal, Optional
from dataclasses import dataclass
from functools import partial

import gradio as gr
from gradio.components.base import Component


from backend.generate_song_cover import (
    get_named_song_dirs,
    get_song_cover_name,
)

progress_bar = gr.Progress()


@dataclass
class EventArgs:
    fn: Callable
    inputs: Optional[list[Component]] = None
    outputs: Optional[list[Component]] = None
    name: Literal["click", "success", "then"] = "success"
    show_progress: Literal["full", "minimal", "hidden"] = "full"


def setup_consecutive_event_listeners(
    component: Component, event_args_list: list[EventArgs]
):
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


def exception_harness(fun, *args, **kwargs):
    new_kwargs = kwargs | {"progress": progress_bar}
    try:
        return fun(*args, **new_kwargs)
    except Exception as e:
        raise gr.Error(str(e))


def confirmation_harness(fun, confirm, *args, **kwargs):
    if confirm:
        return exception_harness(fun, *args, **kwargs)
    else:
        raise gr.Error("Confirmation missing!")


def confirm_box_js(msg):
    formatted_msg = f"'{msg}'"
    return f"(x) => confirm({formatted_msg})"


def update_dropdowns(fn, num_components, value=None, value_indices=[], **kwargs):
    if len(value_indices) != len(set(value_indices)):
        raise ValueError("Value indices must be unique.")
    if value_indices and max(value_indices) >= num_components:
        raise ValueError(
            "Index of a component to update value for exceeds number of components."
        )
    updated_choices = fn(**kwargs)
    update_args = [{"choices": updated_choices} for _ in range(num_components)]
    for index in value_indices:
        update_args[index]["value"] = value
    return tuple(gr.Dropdown(**update_arg) for update_arg in update_args)


def update_cached_input_songs(num_components, value=None, value_indices=[]):
    return update_dropdowns(
        get_named_song_dirs,
        num_components,
        value=value,
        value_indices=value_indices,
    )


def toggle_visible_component(num_components, visible_index):
    update_args = [{"visible": False, "value": None} for _ in range(num_components)]
    update_args[visible_index]["visible"] = True
    return tuple(gr.update(**update_arg) for update_arg in update_args)


def toggle_component_interactivity(num_components, interactive):
    return tuple(gr.update(interactive=interactive) for _ in range(num_components))


def show_hop_slider(pitch_detection_algo):
    if pitch_detection_algo == "mangio-crepe":
        return gr.update(visible=True)
    else:
        return gr.update(visible=False)


def get_song_cover_name_harness(mixed_vocals, song_dir, voice_model=None):
    update_args = {}
    if mixed_vocals or song_dir or voice_model:
        name = exception_harness(
            get_song_cover_name, mixed_vocals, song_dir, voice_model
        )
        update_args["value"] = name
    else:
        update_args["value"] = None
    return gr.update(**update_args)


def setup_consecutive_event_listeners_with_toggled_interactivity(
    component: Component,
    event_args_list: list[EventArgs],
    toggled_components: list[Component],
):
    if len(event_args_list) == 0:
        raise ValueError("Event args list must not be empty.")

    disable_event_args = EventArgs(
        partial(toggle_component_interactivity, len(toggled_components), False),
        outputs=toggled_components,
        name="click",
        show_progress="hidden",
    )
    enable_event_args = EventArgs(
        partial(toggle_component_interactivity, len(toggled_components), True),
        outputs=toggled_components,
        name="then",
        show_progress="hidden",
    )
    event_args_list_augmented = (
        [disable_event_args] + event_args_list + [enable_event_args]
    )
    return setup_consecutive_event_listeners(component, event_args_list_augmented)
