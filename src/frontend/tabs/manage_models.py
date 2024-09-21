"""Module which defines the code for the "Manage models" tab."""

from collections.abc import Sequence
from functools import partial

import gradio as gr
import pandas as pd

from backend.manage_models import (
    delete_all_models,
    delete_models,
    download_model,
    filter_public_models_table,
    get_public_model_tags,
    get_saved_model_names,
    upload_model,
)

from frontend.common import (
    PROGRESS_BAR,
    confirm_box_js,
    confirmation_harness,
    exception_harness,
    identity,
    render_msg,
    update_dropdowns,
)
from frontend.typing_extra import DropdownValue


def _update_models(
    num_components: int,
    value: DropdownValue = None,
    value_indices: Sequence[int] = [],
) -> gr.Dropdown | tuple[gr.Dropdown, ...]:
    """
    Update the choices of one or more dropdown components to the set of
    currently saved voice models.

    Optionally updates the default value of one or more of these
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
    gr.Dropdown | tuple[gr.Dropdown, ...]
        Updated dropdown component or components.

    """
    return update_dropdowns(get_saved_model_names, num_components, value, value_indices)


def _filter_public_models_table(
    tags: Sequence[str],
    query: str,
    progress_bar: gr.Progress,
) -> gr.Dataframe:
    """
    Filter table containing metadata of public voice models by tags and
    a search query.

    Parameters
    ----------
    tags : Sequence[str]
        Tags to filter the metadata table by.
    query : str
        Search query to filter the metadata table by.
    progress_bar : gr.Progress
        Progress bar to display progress.

    Returns
    -------
    gr.Dataframe
        The filtered table rendered in a Gradio dataframe.

    """
    models_table = filter_public_models_table(tags, query, progress_bar)
    return gr.Dataframe(value=models_table)


def _autofill_model_name_and_url(
    public_models_table: pd.DataFrame,
    select_event: gr.SelectData,
) -> tuple[gr.Textbox, gr.Textbox]:
    """
    Autofill two textboxes with respectively the name and URL that is
    saved in the currently selected row of the public models table.

    Parameters
    ----------
    public_models_table : pd.DataFrame
        The public models table saved in a Pandas dataframe.
    select_event : gr.SelectData
        Event containing the index of the currently selected row in the
        public models table.

    Returns
    -------
    name : gr.Textbox
        The textbox containing the model name.

    url : gr.Textbox
        The textbox containing the model URL.

    Raises
    ------
    TypeError
        If the index in the provided event is not a sequence.

    """
    event_index = select_event.index
    if not isinstance(event_index, Sequence):
        err_msg = (
            f"Expected a sequence of indices but got {type(event_index)} from the"
            " provided event."
        )
        raise TypeError(err_msg)
    event_index = event_index[0]
    url = public_models_table.loc[event_index, "URL"]
    name = public_models_table.loc[event_index, "Name"]
    if isinstance(url, str) and isinstance(name, str):
        return gr.Textbox(value=name), gr.Textbox(value=url)
    err_msg = (
        "Expected model name and URL to be strings but got"
        f" {type(name)} and {type(url)} respectively."
    )
    raise TypeError(err_msg)


def render(
    dummy_checkbox: gr.Checkbox,
    confirmation: gr.State,
    model_delete: gr.Dropdown,
    model_1click: gr.Dropdown,
    model_multi: gr.Dropdown,
) -> None:
    """

    Render "Manage models" tab.

    Parameters
    ----------
    dummy_checkbox : gr.Checkbox
        Dummy checkbox component needed for deletion confirmation in the
        "Delete audio" tab and the "Delete models" tab.
    confirmation : gr.State
        Component storing deletion confirmation status in the
        "Delete audio" tab and the "Delete models" tab.
    model_delete : gr.Dropdown
        Dropdown for selecting voice models to delete in the
        "Delete models" tab.
    model_1click : gr.Dropdown
        Dropdown for selecting a voice model to use in the
        "One-click generation" tab.
    model_multi : gr.Dropdown
        Dropdown for selecting a voice model to use in the
        "Multi-step generation" tab.

    """
    # Download tab
    with gr.Tab("Download model"):

        with gr.Accordion("View public models table", open=False):

            gr.Markdown("")
            gr.Markdown("*HOW TO USE*")
            gr.Markdown(
                "- Filter voice models by selecting one or more tags and/or providing a"
                " search query.",
            )
            gr.Markdown(
                "- Select a row in the table to autofill the name and"
                " URL for the given voice model in the form fields below.",
            )
            gr.Markdown("")
            with gr.Row(equal_height=False):
                tags = gr.CheckboxGroup(
                    value=[],
                    label="Tags",
                    choices=get_public_model_tags(),
                )
                search_query = gr.Textbox(label="Search query")
            with gr.Row():
                public_models_table = gr.Dataframe(
                    value=partial(
                        exception_harness(_filter_public_models_table),
                        progress_bar=PROGRESS_BAR,
                    ),
                    inputs=[tags, search_query],
                    headers=["Name", "Description", "Tags", "Credit", "Added", "URL"],
                    label="Public models table",
                    interactive=False,
                )

        with gr.Row():
            model_url = gr.Textbox(
                label="Model URL",
                info=(
                    "Should point to a zip file containing a .pth model file and"
                    " optionally also an .index file."
                ),
            )
            model_name = gr.Textbox(
                label="Model name",
                info="Enter a unique name for the voice model.",
            )

        with gr.Row():
            download_btn = gr.Button("Download 🌐", variant="primary", scale=19)
            download_msg = gr.Textbox(
                label="Output message",
                interactive=False,
                scale=20,
            )

        public_models_table.select(
            _autofill_model_name_and_url,
            inputs=public_models_table,
            outputs=[model_name, model_url],
            show_progress="hidden",
        )

        download_btn_click = download_btn.click(
            partial(
                exception_harness(download_model),
                progress_bar=PROGRESS_BAR,
            ),
            inputs=[model_url, model_name],
            outputs=download_msg,
        ).success(
            partial(
                render_msg,
                "[+] Succesfully downloaded voice model!",
            ),
            inputs=model_name,
            outputs=download_msg,
            show_progress="hidden",
        )

    # Upload tab
    with gr.Tab("Upload model"):
        with gr.Accordion("HOW TO USE"):
            gr.Markdown("")
            gr.Markdown(
                "1. Find the .pth file for a locally trained RVC model (e.g. in your"
                " local weights folder) and optionally also a corresponding .index file"
                " (e.g. in your logs/[name] folder)",
            )
            gr.Markdown(
                "2. Upload the files directly or save them to a folder, then compress"
                " that folder and upload the resulting .zip file",
            )
            gr.Markdown("3. Enter a unique name for the uploaded model")
            gr.Markdown("4. Click 'Upload'")

        with gr.Row(equal_height=False):
            model_files = gr.File(
                label="Files",
                file_count="multiple",
                file_types=[".zip", ".pth", ".index"],
            )

            local_model_name = gr.Textbox(label="Model name")

        with gr.Row():
            upload_btn = gr.Button("Upload", variant="primary", scale=19)
            upload_msg = gr.Textbox(
                label="Output message",
                interactive=False,
                scale=20,
            )
            upload_btn_click = upload_btn.click(
                partial(exception_harness(upload_model), progress_bar=PROGRESS_BAR),
                inputs=[model_files, local_model_name],
                outputs=upload_msg,
            ).success(
                partial(
                    render_msg,
                    "[+] Successfully uploaded voice model!",
                ),
                inputs=local_model_name,
                outputs=upload_msg,
                show_progress="hidden",
            )

    with gr.Tab("Delete models"):
        with gr.Row():
            with gr.Column():
                model_delete.render()
                delete_btn = gr.Button("Delete selected", variant="secondary")
                delete_all_btn = gr.Button("Delete all", variant="primary")
            with gr.Column():
                delete_msg = gr.Textbox(label="Output message", interactive=False)
        delete_btn_click = (
            delete_btn.click(
                # NOTE not sure why, but in order for subsequent event
                # listener to trigger, changes coming from the js code
                # have to be routed through an identity function which
                # takes as input some dummy component of type bool.
                identity,
                inputs=dummy_checkbox,
                outputs=confirmation,
                js=confirm_box_js(
                    "Are you sure you want to delete the selected voice models?",
                ),
                show_progress="hidden",
            )
            .then(
                partial(confirmation_harness(delete_models), progress_bar=PROGRESS_BAR),
                inputs=[confirmation, model_delete],
                outputs=delete_msg,
            )
            .success(
                partial(render_msg, "[-] Successfully deleted selected voice models!"),
                outputs=delete_msg,
                show_progress="hidden",
            )
        )

        delete_all_btn_click = (
            delete_all_btn.click(
                identity,
                inputs=dummy_checkbox,
                outputs=confirmation,
                js=confirm_box_js("Are you sure you want to delete all voice models?"),
                show_progress="hidden",
            )
            .then(
                partial(
                    confirmation_harness(delete_all_models),
                    progress_bar=PROGRESS_BAR,
                ),
                inputs=confirmation,
                outputs=delete_msg,
            )
            .success(
                partial(render_msg, "[-] Successfully deleted all voice models!"),
                outputs=delete_msg,
                show_progress="hidden",
            )
        )

    for click_event in [
        download_btn_click,
        upload_btn_click,
        delete_btn_click,
        delete_all_btn_click,
    ]:
        click_event.success(
            partial(_update_models, 3, [], [2]),
            outputs=[model_1click, model_multi, model_delete],
            show_progress="hidden",
        )
