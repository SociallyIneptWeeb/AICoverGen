"""
Module which defines initialization of the core components of the
Ultimate RVC project.
"""

from pathlib import Path

from rich import print as rprint

import requests

from common import RVC_MODELS_DIR
from typing_extra import SeparationModel, StrPath

from core.generate_song_cover import AUDIO_SEPARATOR
from core.manage_models import download_model

RVC_DOWNLOAD_URL = "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/"


def _download_base_model(url: str, name: str, directory: StrPath) -> None:
    """
    Download a base model and save it to a directory.

    Parameters
    ----------
    url : str
        An URL pointing to a location where a base model is hosted.
    name : str
        The name of the base model to download.
    directory : str
        The path to the directory where the base model should be saved.

    """
    dir_path = Path(directory)
    with requests.get(f"{url}{name}", timeout=10) as r:
        r.raise_for_status()
        with (dir_path / name).open("wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)


def initialize() -> None:
    """Initialize the core components of the Ultimate RVC project."""
    for separator_model in SeparationModel:
        rprint(f"Downloading {separator_model}...")
        AUDIO_SEPARATOR.download_model_files(separator_model)

    base_model_names = ["hubert_base.pt", "rmvpe.pt"]
    for base_model_name in base_model_names:
        rprint(f"Downloading {base_model_name}...")
        _download_base_model(RVC_DOWNLOAD_URL, base_model_name, RVC_MODELS_DIR)

    named_model_links = [
        (
            "https://huggingface.co/damnedraxx/TaylorSwift/resolve/main/TaylorSwift.zip",
            "Taylor Swift",
        ),
        (
            "https://huggingface.co/Vermiculos/balladjames/resolve/main/Ballad%20James.zip?download=true",
            "James Hetfield",
        ),
        ("https://huggingface.co/ryolez/MMLP/resolve/main/MMLP.zip", "Eminem"),
    ]
    for model_url, model_name in named_model_links:
        rprint(f"Downloading {model_name}...")
        try:
            download_model(model_url, model_name)
        except Exception as e:
            rprint(f"Failed to download {model_name}: {e}")
    rprint("All models downloaded!")