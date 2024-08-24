"""
This module contains functions to manage voice models.
"""

from typings.extra import ModelsTable, ModelsTablePredicate

import os
import re
import shutil
import urllib.request
import zipfile

import gradio as gr

from backend.common import copy_files_to_new_folder, display_progress, json_load
from backend.exceptions import (
    FileTypeError,
    InputMissingError,
    PathExistsError,
    PathNotFoundError,
)

from common import RVC_MODELS_DIR

PUBLIC_MODELS = json_load(os.path.join(RVC_MODELS_DIR, "public_models.json"))


def get_current_models() -> list[str]:
    """
    Get the names of all saved voice models.

    Returns
    -------
    list[str]
        A list of names of all saved voice models.
    """
    models_list = os.listdir(RVC_MODELS_DIR)
    items_to_remove = ["hubert_base.pt", "MODELS.txt", "public_models.json", "rmvpe.pt"]
    return [item for item in models_list if item not in items_to_remove]


def load_public_models_table(
    predicates: list[ModelsTablePredicate],
    progress_bar: gr.Progress | None = None,
    percentage: float = 0.0,
) -> ModelsTable:
    """
    Load the public models table and filter it by the given predicates.

    Parameters
    ----------
    predicates : list[ModelsTablePredicate]
        List of predicates to filter the models table by.
    progress_bar : gr.Progress, optional
        Gradio progress bar to update.
    percentage : float, default=0.0
        Percentage to display in the progress bar.

    Returns
    -------
    ModelsTable
        The public models table, filtered by the given predicates.
    """
    models_table: ModelsTable = []
    keys = ["name", "description", "tags", "credit", "added", "url"]
    display_progress("[~] Loading public models table ...", percentage, progress_bar)
    for model in PUBLIC_MODELS["voice_models"]:
        if all([predicate(model) for predicate in predicates]):
            models_table.append([model[key] for key in keys])

    return models_table


def load_public_model_tags() -> list[str]:
    """
    Load the tags of all public voice models.

    Returns
    -------
    list[str]
        A list of all tags of public voice models.
    """
    return list(PUBLIC_MODELS["tags"].keys())


def filter_public_models_table(
    tags: list[str],
    query: str,
    progress_bar: gr.Progress | None = None,
    percentage: float = 0.0,
) -> ModelsTable:
    """
    Filter the public models table by a set of tags and a search query.

    The search query is matched against the name, description, tags, credit,
    and added date of each model in the public models table.
    Case insensitive search is performed.
    If the search query is empty, the models table is filtered only by the tags.

    Parameters
    ----------
    tags : list[str]
        List of tags to filter the models table by.
    query : str
        Search query to filter the models table by.
    progress_bar : gr.Progress, optional
        Gradio progress bar to update.
    percentage : float, default=0.0
        Percentage to display in the progress bar.

    Returns
    -------
    ModelsTable
        The public models table, filtered by the given tags and the given query.
    """
    tags_predicate: ModelsTablePredicate = lambda model: all(
        tag in model["tags"] for tag in tags
    )
    query_predicate: ModelsTablePredicate = lambda model: (
        query.lower()
        in f"{model['name']} {model['description']} {' '.join(model['tags'])} {model['credit']} {model['added']}"
        .lower()
        if query
        else True
    )

    filter_fns = [tags_predicate, query_predicate]

    return load_public_models_table(filter_fns, progress_bar, percentage)


def _extract_model_zip(extraction_folder: str, zip_name: str, remove_zip: bool) -> None:
    """
    Extract a voice model zip file to a directory.

    Parameters
    ----------
    extraction_folder : str
        The directory to extract the voice model to.
    zip_name : str
        The name of the zip file to extract.
    remove_zip : bool
        Whether to remove the zip file after extraction.

    Raises
    ------
    PathNotFoundError
        If no .pth model file is found in the extracted zip folder.
    """
    try:
        os.makedirs(extraction_folder)
        with zipfile.ZipFile(zip_name, "r") as zip_ref:
            zip_ref.extractall(extraction_folder)

        index_filepath, model_filepath = None, None
        for root, _, files in os.walk(extraction_folder):
            for name in files:
                if (
                    name.endswith(".index")
                    and os.stat(os.path.join(root, name)).st_size > 1024 * 100
                ):
                    index_filepath = os.path.join(root, name)

                if (
                    name.endswith(".pth")
                    and os.stat(os.path.join(root, name)).st_size > 1024 * 1024 * 40
                ):
                    model_filepath = os.path.join(root, name)

        if not model_filepath:
            raise PathNotFoundError(
                "No .pth model file was found in the extracted zip folder."
            )
        # move model and index file to extraction folder

        os.rename(
            model_filepath,
            os.path.join(extraction_folder, os.path.basename(model_filepath)),
        )
        if index_filepath:
            os.rename(
                index_filepath,
                os.path.join(extraction_folder, os.path.basename(index_filepath)),
            )

        # remove any unnecessary nested folders
        for filepath in os.listdir(extraction_folder):
            if os.path.isdir(os.path.join(extraction_folder, filepath)):
                shutil.rmtree(os.path.join(extraction_folder, filepath))

    except Exception as e:
        if os.path.isdir(extraction_folder):
            shutil.rmtree(extraction_folder)
        raise e
    finally:
        if remove_zip and os.path.exists(zip_name):
            os.remove(zip_name)


def download_online_model(
    url: str,
    dir_name: str,
    progress_bar: gr.Progress | None = None,
    percentages: tuple[float, float] = (0.0, 0.5),
) -> str:
    """
    Download a voice model from a given URL and extract it to a directory.

    Parameters
    ----------
    url : str
        The URL of the voice model to download.
    dir_name : str
        The name of the directory to extract the voice model to.
    progress_bar : gr.Progress, optional
        Gradio progress bar to update.
    percentages : tuple[float, float], default=(0.0, 0.5)
        Percentages to display in the progress bar.

    Returns
    -------
    str
        Success message.

    Raises
    ------
    InputMissingError
        If an URL or a voice model directory name is not given.
    PathExistsError
        If the voice model directory already exists.
    """
    if not url:
        raise InputMissingError("Download link to model missing!")
    if not dir_name:
        raise InputMissingError("Model name missing!")
    extraction_folder = os.path.join(RVC_MODELS_DIR, dir_name)
    if os.path.exists(extraction_folder):
        raise PathExistsError(
            f'Voice model directory "{dir_name}" already exists! Choose a different'
            " name for your voice model."
        )
    zip_name = url.split("/")[-1].split("?")[0]

    # NOTE in case huggingface link is a direct link rather
    # than a resolve link then convert it to a resolve link
    url = re.sub(
        r"https://huggingface.co/([^/]+)/([^/]+)/blob/(.*)",
        r"https://huggingface.co/\1/\2/resolve/\3",
        url,
    )
    if "pixeldrain.com" in url:
        url = f"https://pixeldrain.com/api/file/{zip_name}"

    display_progress(
        f"[~] Downloading voice model with name '{dir_name}'...",
        percentages[0],
        progress_bar,
    )

    urllib.request.urlretrieve(url, zip_name)

    display_progress("[~] Extracting zip file...", percentages[1], progress_bar)

    _extract_model_zip(extraction_folder, zip_name, remove_zip=True)
    return f"[+] Model with name '{dir_name}' successfully downloaded!"


def upload_local_model(
    input_paths: list[str],
    dir_name: str,
    progress_bar: gr.Progress | None = None,
    percentage: float = 0.0,
) -> str:
    """
    Upload a voice model from either a local zip file or a local .pth file
    and an optional index file.

    Parameters
    ----------
    input_paths : list[str]
        Paths of the local files to upload.
    dir_name : str
        The name of the directory to save the voice model files in.
    progress_bar : gr.Progress, optional
        Gradio progress bar to update.
    percentage : float, default=0.0
        Percentage to display in the progress bar.

    Returns
    -------
    str
        Success message.

    Raises
    ------
    InputMissingError
        If no file paths or no voice model directory name is given.
    ValueError
        If more than two file paths are given.
    PathExistsError
        If a voice model directory by the given name already exists.
    FileTypeError
        If a single uploaded file is not a .pth file or a .zip file.
        If two uploaded files are not a .pth file and an .index file.
    """
    if not input_paths:
        raise InputMissingError("No files selected!")
    if len(input_paths) > 2:
        raise ValueError("At most two files can be uploaded!")
    if not dir_name:
        raise InputMissingError("Model name missing!")
    output_folder = os.path.join(RVC_MODELS_DIR, dir_name)
    if os.path.exists(output_folder):
        raise PathExistsError(
            f'Voice model directory "{dir_name}" already exists! Choose a different'
            " name for your voice model."
        )
    if len(input_paths) == 1:
        input_path = input_paths[0]
        if os.path.splitext(input_path)[1] == ".pth":
            display_progress("[~] Copying .pth file ...", percentage, progress_bar)
            copy_files_to_new_folder(input_paths, output_folder)
        # NOTE a .pth file is actually itself a zip file
        elif zipfile.is_zipfile(input_path):
            display_progress("[~] Extracting zip file...", percentage, progress_bar)
            _extract_model_zip(output_folder, input_path, remove_zip=False)
        else:
            raise FileTypeError(
                "Only a .pth file or a .zip file can be uploaded by itself!"
            )
    else:
        # sort two input files by extension type
        input_names_sorted = sorted(input_paths, key=lambda f: os.path.splitext(f)[1])
        index_name, pth_name = input_names_sorted
        if (
            os.path.splitext(pth_name)[1] == ".pth"
            and os.path.splitext(index_name)[1] == ".index"
        ):
            display_progress(
                "[~] Copying .pth file and index file ...", percentage, progress_bar
            )
            copy_files_to_new_folder(input_paths, output_folder)
        else:
            raise FileTypeError(
                "Only a .pth file and an .index file can be uploaded together!"
            )

    return f"[+] Model with name '{dir_name}' successfully uploaded!"


def delete_models(
    model_names: list[str],
    progress_bar: gr.Progress | None = None,
    percentage: float = 0.0,
) -> str:
    """
    Delete one or more voice models.

    Parameters
    ----------
    model_names : list[str]
        Names of the models to delete.
    progress_bar : gr.Progress, optional
        Gradio progress bar to update.
    percentage : float, default=0.0
        Percentage to display in the progress bar.

    Returns
    -------
    str
        Success message.

    Raises
    ------
    InputMissingError
        If no model names are given.
    PathNotFoundError
        If a voice model directory does not exist.
    """
    if not model_names:
        raise InputMissingError("No models selected!")
    display_progress("[~] Deleting selected models ...", percentage, progress_bar)
    for model_name in model_names:
        model_dir = os.path.join(RVC_MODELS_DIR, model_name)
        if not os.path.isdir(model_dir):
            raise PathNotFoundError(
                f'Voice model directory "{model_name}" does not exist!'
            )
        shutil.rmtree(model_dir)
    models_names_formatted = [f"'{w}'" for w in model_names]
    if len(model_names) == 1:
        return f"[+] Model with name {models_names_formatted[0]} successfully deleted!"
    else:
        first_models = ", ".join(models_names_formatted[:-1])
        last_model = models_names_formatted[-1]
        return (
            f"[+] Models with names {first_models} and {last_model} successfully"
            " deleted!"
        )


def delete_all_models(
    progress_bar: gr.Progress | None = None, percentage: float = 0.0
) -> str:
    """
    Delete all voice models.

    Parameters
    ----------
    progress_bar : gr.Progress, optional
        Gradio progress bar to update.
    percentage : float, default=0.0
        Percentage to display in the progress bar.

    Returns
    -------
    str
        Success message.
    """
    all_models = get_current_models()
    display_progress("[~] Deleting all models ...", percentage, progress_bar)
    for model_name in all_models:
        model_dir = os.path.join(RVC_MODELS_DIR, model_name)
        if os.path.isdir(model_dir):
            shutil.rmtree(model_dir)
    return "[+] All models successfully deleted!"
