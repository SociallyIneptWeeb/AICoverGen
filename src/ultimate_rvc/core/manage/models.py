"""Module which defines functions to manage voice models."""

import re
import shutil
import urllib.request
import zipfile
from _collections_abc import Sequence
from pathlib import Path

import gradio as gr

from ultimate_rvc.common import RVC_MODELS_DIR
from ultimate_rvc.core.common import (
    copy_files_to_new_dir,
    display_progress,
    json_load,
    validate_url,
)
from ultimate_rvc.core.exceptions import (
    Entity,
    Location,
    NotFoundError,
    NotProvidedError,
    UIMessage,
    UploadFormatError,
    UploadLimitError,
    VoiceModelExistsError,
    VoiceModelNotFoundError,
)
from ultimate_rvc.core.typing_extra import (
    ModelMetaData,
    ModelMetaDataList,
    ModelMetaDataPredicate,
    ModelMetaDataTable,
    ModelTagName,
)
from ultimate_rvc.typing_extra import StrPath

PUBLIC_MODELS_JSON = json_load(Path(__file__).parent / "public_models.json")
PUBLIC_MODELS_TABLE = ModelMetaDataTable.model_validate(PUBLIC_MODELS_JSON)


def _extract_model(
    zip_file: StrPath,
    extraction_dir: StrPath,
    remove_incomplete: bool = True,
    remove_zip: bool = False,
) -> None:
    """
    Extract a zipped voice model to a directory.

    Parameters
    ----------
    zip_file : StrPath
        The path to a zip file containing the voice model to extract.
    extraction_dir : StrPath
        The path to the directory to extract the voice model to.

    remove_incomplete : bool, default=True
        Whether to remove the extraction directory if the extraction
        process fails.
    remove_zip : bool, default=False
        Whether to remove the zip file once the extraction process is
        complete.

    Raises
    ------
    NotFoundError
        If no model file is found in the extracted zip file.

    """
    extraction_path = Path(extraction_dir)
    zip_path = Path(zip_file)
    extraction_completed = False
    try:
        extraction_path.mkdir(parents=True)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extraction_path)
        file_path_map = {
            ext: Path(root, name)
            for root, _, files in extraction_path.walk()
            for name in files
            for ext in [".index", ".pth"]
            if Path(name).suffix == ext
            and Path(root, name).stat().st_size
            > 1024 * (100 if ext == ".index" else 1024 * 40)
        }
        if ".pth" not in file_path_map:
            raise NotFoundError(
                entity=Entity.MODEL_FILE,
                location=Location.EXTRACTED_ZIP_FILE,
                is_path=False,
            )

        # move model and index file to root of the extraction directory
        for file_path in file_path_map.values():
            file_path.rename(extraction_path / file_path.name)

        # remove any sub-directories within the extraction directory
        for path in extraction_path.iterdir():
            if path.is_dir():
                shutil.rmtree(path)
        extraction_completed = True
    finally:
        if not extraction_completed and remove_incomplete and extraction_path.is_dir():
            shutil.rmtree(extraction_path)
        if remove_zip and zip_path.exists():
            zip_path.unlink()


def get_saved_model_names() -> list[str]:
    """
    Get the names of all saved voice models.

    Returns
    -------
    list[str]
        A list of names of all saved voice models.

    """
    model_paths = RVC_MODELS_DIR.iterdir()
    names_to_remove = ["hubert_base.pt", "rmvpe.pt"]
    return [
        model_path.name
        for model_path in model_paths
        if model_path.name not in names_to_remove
    ]


def load_public_models_table(
    predicates: Sequence[ModelMetaDataPredicate],
    progress_bar: gr.Progress | None = None,
    percentage: float = 0.5,
) -> ModelMetaDataList:
    """
    Load table containing metadata of public voice models, optionally
    filtered by a set of predicates.

    Parameters
    ----------
    predicates : Sequence[ModelMetaDataPredicate]
        Predicates to filter the metadata table by.
    progress_bar : gr.Progress, optional
        Gradio progress bar to update.
    percentage : float, default=0.5
        Percentage to display in the progress bar.

    Returns
    -------
    ModelMetaDataList
        List containing metadata for each public voice model that
        satisfies the given predicates.

    """
    display_progress("[~] Loading public models table ...", percentage, progress_bar)
    return [
        [
            model.name,
            model.description,
            model.tags,
            model.credit,
            model.added,
            model.url,
        ]
        for model in PUBLIC_MODELS_TABLE.models
        if all(predicate(model) for predicate in predicates)
    ]


def get_public_model_tags() -> list[ModelTagName]:
    """
    get the names of all valid public voice model tags.

    Returns
    -------
    list[str]
        A list of names of all valid public voice model tags.

    """
    return [tag.name for tag in PUBLIC_MODELS_TABLE.tags]


def filter_public_models_table(
    tags: Sequence[str],
    query: str,
    progress_bar: gr.Progress | None = None,
    percentage: float = 0.5,
) -> ModelMetaDataList:
    """
    Filter table containing metadata of public voice models by tags and
    a search query.


    The search query is matched against the name, description, tags,
    credit,and added date of each entry in the metadata table. Case
    insensitive search is performed. If the search query is empty, the
    metadata table is filtered only bythe given tags.

    Parameters
    ----------
    tags : Sequence[str]
        Tags to filter the metadata table by.
    query : str
        Search query to filter the metadata table by.
    progress_bar : gr.Progress, optional
        Gradio progress bar to update.
    percentage : float, default=0.5
        Percentage to display in the progress bar.

    Returns
    -------
    ModelMetaDataList
        List containing metadata for each public voice model that
        match the given tags and search query.

    """

    def _tags_predicate(model: ModelMetaData) -> bool:
        return all(tag in model.tags for tag in tags)

    def _query_predicate(model: ModelMetaData) -> bool:
        return (
            query.lower()
            in (
                f"{model.name} {model.description} {' '.join(model.tags)} "
                f"{model.credit} {model.added}"
            ).lower()
            if query
            else True
        )

    filter_fns = [_tags_predicate, _query_predicate]

    return load_public_models_table(filter_fns, progress_bar, percentage)


def download_model(
    url: str,
    name: str,
    progress_bar: gr.Progress | None = None,
    percentages: tuple[float, float] = (0.0, 0.5),
) -> None:
    """
    Download a zipped voice model.

    Parameters
    ----------
    url : str
        An URL pointing to a location where the zipped voice model can
        be downloaded from.
    name : str
        The name to give to the downloaded voice model.
    progress_bar : gr.Progress, optional
        Gradio progress bar to update.
    percentages : tuple[float, float], default=(0.0, 0.5)
        Percentages to display in the progress bar.

    Raises
    ------
    NotProvidedError
        If no URL or name is provided.
    VoiceModelExistsError
        If a voice model with the provided name already exists.

    """
    if not url:
        raise NotProvidedError(entity=Entity.URL)
    if not name:
        raise NotProvidedError(entity=Entity.MODEL_NAME)
    extraction_path = RVC_MODELS_DIR / name
    if extraction_path.exists():
        raise VoiceModelExistsError(name)

    validate_url(url)
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
        "[~] Downloading voice model ...",
        percentages[0],
        progress_bar,
    )
    urllib.request.urlretrieve(url, zip_name)  # noqa: S310

    display_progress("[~] Extracting zip file...", percentages[1], progress_bar)
    _extract_model(zip_name, extraction_path, remove_zip=True)


def upload_model(
    files: Sequence[StrPath],
    name: str,
    progress_bar: gr.Progress | None = None,
    percentage: float = 0.5,
) -> None:
    """
    Upload a voice model from either a zip file or a .pth file and an
    optional index file.

    Parameters
    ----------
    files : Sequence[StrPath]
        Paths to the files to upload.
    name : str
        The name to give to the uploaded voice model.
    progress_bar : gr.Progress, optional
        Gradio progress bar to update.
    percentage : float, default=0.5
        Percentage to display in the progress bar.

    Raises
    ------
    NotProvidedError
        If no file paths or name are provided.
    VoiceModelExistsError
        If a voice model with the provided name already
        exists.
    UploadFormatError
        If a single uploaded file is not a .pth file or a .zip file.
        If two uploaded files are not a .pth file and an .index file.
    UploadLimitError
        If more than two file paths are provided.

    """
    if not files:
        raise NotProvidedError(entity=Entity.FILES, ui_msg=UIMessage.NO_UPLOADED_FILES)
    if not name:
        raise NotProvidedError(entity=Entity.MODEL_NAME)
    model_dir_path = RVC_MODELS_DIR / name
    if model_dir_path.exists():
        raise VoiceModelExistsError(name)
    sorted_file_paths = sorted([Path(f) for f in files], key=lambda f: f.suffix)
    match sorted_file_paths:
        case [file_path]:
            if file_path.suffix == ".pth":
                display_progress("[~] Copying .pth file ...", percentage, progress_bar)
                copy_files_to_new_dir([file_path], model_dir_path)
            # NOTE a .pth file is actually itself a zip file
            elif zipfile.is_zipfile(file_path):
                display_progress("[~] Extracting zip file...", percentage, progress_bar)
                _extract_model(file_path, model_dir_path)
            else:
                raise UploadFormatError(
                    entity=Entity.FILES,
                    formats=[".pth", ".zip"],
                    multiple=False,
                )
        case [index_path, pth_path]:
            if index_path.suffix == ".index" and pth_path.suffix == ".pth":
                display_progress(
                    "[~] Copying .pth file and index file ...",
                    percentage,
                    progress_bar,
                )
                copy_files_to_new_dir([index_path, pth_path], model_dir_path)
            else:
                raise UploadFormatError(
                    entity=Entity.FILES,
                    formats=[".pth", ".index"],
                    multiple=True,
                )
        case _:
            raise UploadLimitError(entity=Entity.FILES, limit="two")


def delete_models(
    names: Sequence[str],
    progress_bar: gr.Progress | None = None,
    percentage: float = 0.5,
) -> None:
    """
    Delete one or more voice models.

    Parameters
    ----------
    names : Sequence[str]
        Names of the voice models to delete.
    progress_bar : gr.Progress, optional
        Gradio progress bar to update.
    percentage : float, default=0.5
        Percentage to display in the progress bar.

    Raises
    ------
    NotProvidedError
        If no names are provided.
    VoiceModelNotFoundError
        If a voice model with a provided name does not exist.

    """
    if not names:
        raise NotProvidedError(
            entity=Entity.MODEL_NAMES,
            ui_msg=UIMessage.NO_VOICE_MODELS,
        )
    display_progress(
        "[~] Deleting voice models ...",
        percentage,
        progress_bar,
    )
    for name in names:
        model_dir_path = RVC_MODELS_DIR / name
        if not model_dir_path.is_dir():
            raise VoiceModelNotFoundError(name)
        shutil.rmtree(model_dir_path)


def delete_all_models(
    progress_bar: gr.Progress | None = None,
    percentage: float = 0.5,
) -> None:
    """
    Delete all voice models.

    Parameters
    ----------
    progress_bar : gr.Progress, optional
        Gradio progress bar to update.
    percentage : float, default=0.5
        Percentage to display in the progress bar.

    """
    all_model_names = get_saved_model_names()
    display_progress("[~] Deleting all voice models ...", percentage, progress_bar)
    for model_name in all_model_names:
        model_dir_path = RVC_MODELS_DIR / model_name
        if model_dir_path.is_dir():
            shutil.rmtree(model_dir_path)
