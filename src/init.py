"""
Script which downloads the models required for running the Ultimate
RVC app.
"""

from logging import WARNING
from pathlib import Path

from rich import print as rprint

import requests

from audio_separator.separator import Separator

from common import RVC_MODELS_DIR, SEPARATOR_MODELS_DIR
from typing_extra import SeparationModel, StrPath

from backend.manage_models import download_model

RVC_DOWNLOAD_URL = "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/"


def download_base_model(url: str, name: str, directory: StrPath) -> None:
    """
    Download a model and save it to a directory.

    Parameters
    ----------
    url : str
        An URL pointing to a location where a model is hosted.
    name : str
        The name of the model to download.
    directory : str
        The path to the directory where the model should be saved.

    """
    dir_path = Path(directory)
    with requests.get(f"{url}{name}", timeout=10) as r:
        r.raise_for_status()
        with (dir_path / name).open("wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)


if __name__ == "__main__":
    separator = Separator(
        log_level=WARNING,
        model_file_dir=SEPARATOR_MODELS_DIR,
    )
    for separator_model in SeparationModel:
        rprint(f"Downloading {separator_model}...")
        separator.download_model_files(separator_model)

    base_model_names = ["hubert_base.pt", "rmvpe.pt"]
    for base_model_name in base_model_names:
        rprint(f"Downloading {base_model_name}...")
        download_base_model(RVC_DOWNLOAD_URL, base_model_name, RVC_MODELS_DIR)

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
