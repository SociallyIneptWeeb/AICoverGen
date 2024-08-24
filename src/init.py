"""
This script downloads the models required for running the Ultimmate RVC app.
"""

import os

import requests

from common import RVC_MODELS_DIR

RVC_DOWNLOAD_LINK = "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/"


def dl_model(link: str, model_name: str, dir_name: str) -> None:
    """
    Download a model from a link and save it to a directory.

    Parameters
    ----------
    link : str
        The link to the site where the model is hosted.
    model_name : str
        The name of the model to download.
    dir_name : str
        The directory to save the model to.
    """
    with requests.get(f"{link}{model_name}") as r:
        r.raise_for_status()
        with open(os.path.join(dir_name, model_name), "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)


if __name__ == "__main__":

    rvc_model_names = ["hubert_base.pt", "rmvpe.pt"]
    for model in rvc_model_names:
        print(f"Downloading {model}...")
        dl_model(RVC_DOWNLOAD_LINK, model, RVC_MODELS_DIR)

    print("All models downloaded!")
