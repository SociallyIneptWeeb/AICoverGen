"""
Script which downloads the models required for running the Ultimate
RVC app.
"""

from pathlib import Path

from rich import print as rprint

import requests

from common import RVC_MODELS_DIR
from typing_extra import StrPath

RVC_DOWNLOAD_URL = "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/"


def download_model(url: str, name: str, directory: StrPath) -> None:
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
    model_names = ["hubert_base.pt", "rmvpe.pt"]
    for model_name in model_names:
        rprint(f"Downloading {model_name}...")
        download_model(RVC_DOWNLOAD_URL, model_name, RVC_MODELS_DIR)

    rprint("All models downloaded!")
