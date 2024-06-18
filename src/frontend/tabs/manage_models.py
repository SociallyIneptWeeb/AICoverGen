from functools import partial

import gradio as gr


from frontend.common import (
    exception_harness,
    confirmation_harness,
    confirm_box_js,
    update_dropdowns,
)

from backend.manage_voice_models import (
    get_current_models,
    load_public_models_table,
    load_public_model_tags,
    filter_public_models_table,
    download_online_model,
    upload_local_model,
    delete_models,
    delete_all_models,
)


def _update_model_lists(num_components, value=None, value_indices=[]):
    return update_dropdowns(
        get_current_models, num_components, value=value, value_indices=value_indices
    )


def _filter_public_models_table_harness(tags, query, progress):
    models_table = filter_public_models_table(tags, query, progress)
    return gr.DataFrame(value=models_table)


def _pub_dl_autofill(pub_models, event: gr.SelectData):
    return gr.Text(value=pub_models.loc[event.index[0], "URL"]), gr.Text(
        value=pub_models.loc[event.index[0], "Model Name"]
    )


def render(
    dummy_deletion_checkbox,
    delete_confirmation,
    rvc_models_to_delete,
    rvc_model,
    rvc_model2,
):

    # Download tab
    with gr.Tab("Download model"):

        with gr.Accordion("View public models table", open=False):

            gr.Markdown("")
            gr.Markdown("HOW TO USE")
            gr.Markdown("- Filter models using tags or search bar")
            gr.Markdown("- Select a row to autofill the download link and model name")

            filter_tags = gr.CheckboxGroup(
                value=[],
                label="Show voice models with tags",
                choices=load_public_model_tags(),
            )
            search_query = gr.Text(label="Search")

            public_models_table = gr.DataFrame(
                value=load_public_models_table([]),
                headers=[
                    "Model Name",
                    "Description",
                    "Tags",
                    "Credit",
                    "Added",
                    "URL",
                ],
                label="Available Public Models",
                interactive=False,
            )

        with gr.Row():
            model_zip_link = gr.Text(
                label="Download link to model",
                info="Should point to a zip file containing a .pth model file and an optional .index file.",
            )
            model_name = gr.Text(
                label="Model name",
                info="Enter a unique name for the model.",
            )

        with gr.Row():
            download_btn = gr.Button("Download 🌐", variant="primary", scale=19)
            dl_output_message = gr.Text(
                label="Output message", interactive=False, scale=20
            )

        download_button_click = download_btn.click(
            partial(exception_harness, download_online_model),
            inputs=[model_zip_link, model_name],
            outputs=dl_output_message,
        )

        public_models_table.select(
            _pub_dl_autofill,
            inputs=public_models_table,
            outputs=[model_zip_link, model_name],
            show_progress="hidden",
        )
        search_query.change(
            partial(exception_harness, _filter_public_models_table_harness),
            inputs=[filter_tags, search_query],
            outputs=public_models_table,
        )
        filter_tags.select(
            partial(exception_harness, _filter_public_models_table_harness),
            inputs=[filter_tags, search_query],
            outputs=public_models_table,
        )

    # Upload tab
    with gr.Tab("Upload model"):
        with gr.Accordion("HOW TO USE"):
            gr.Markdown(
                "- Find locally trained RVC v2 model file (weights folder) and optional index file (logs/[name] folder)"
            )
            gr.Markdown(
                "- Upload model file and optional index file directly or compress into a zip file and upload that"
            )
            gr.Markdown("- Enter a unique name for the model")
            gr.Markdown("- Click 'Upload model'")

        with gr.Row():
            with gr.Column():
                model_files = gr.File(label="Files", file_count="multiple")

            local_model_name = gr.Text(label="Model name")

        with gr.Row():
            model_upload_button = gr.Button("Upload model", variant="primary", scale=19)
            local_upload_output_message = gr.Text(
                label="Output message", interactive=False, scale=20
            )
            model_upload_button_click = model_upload_button.click(
                partial(exception_harness, upload_local_model),
                inputs=[model_files, local_model_name],
                outputs=local_upload_output_message,
            )

    with gr.Tab("Delete models"):
        with gr.Row():
            with gr.Column():
                rvc_models_to_delete.render()
            with gr.Column():
                rvc_models_deleted_message = gr.Text(
                    label="Output message", interactive=False
                )

        with gr.Row():
            with gr.Column():
                delete_models_button = gr.Button(
                    "Delete selected models", variant="secondary"
                )
                delete_all_models_button = gr.Button(
                    "Delete all models", variant="primary"
                )
            with gr.Column():
                pass
        delete_models_button_click = delete_models_button.click(
            # NOTE not sure why, but in order for subsequent event listener
            # to trigger, changes coming from the js code
            # have to be routed through an identity function which takes as
            # input some dummy component of type bool.
            lambda x: x,
            inputs=dummy_deletion_checkbox,
            outputs=delete_confirmation,
            js=confirm_box_js("Are you sure you want to delete the selected models?"),
        ).then(
            partial(confirmation_harness, delete_models),
            inputs=[delete_confirmation, rvc_models_to_delete],
            outputs=rvc_models_deleted_message,
        )

        delete_all_models_btn_click = delete_all_models_button.click(
            lambda x: x,
            inputs=dummy_deletion_checkbox,
            outputs=delete_confirmation,
            js=confirm_box_js("Are you sure you want to delete all models?"),
        ).then(
            partial(confirmation_harness, delete_all_models),
            inputs=delete_confirmation,
            outputs=rvc_models_deleted_message,
        )

    for click_event in [
        download_button_click,
        model_upload_button_click,
        delete_models_button_click,
        delete_all_models_btn_click,
    ]:
        click_event.success(
            partial(_update_model_lists, 3, [], [2]),
            outputs=[rvc_model, rvc_model2, rvc_models_to_delete],
        )
