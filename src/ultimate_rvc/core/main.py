"""
Module which defines functions for initializing the core of the Ultimate
RVC project.
"""

from pathlib import Path

from rich import print as rprint

from ultimate_rvc.common import RVC_MODELS_DIR
from ultimate_rvc.core.common import download_base_models
from ultimate_rvc.core.generate.song_cover import initialize_audio_separator
from ultimate_rvc.core.manage.models import download_model


def download_sample_models() -> None:
    """Download sample RVC models."""
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
        if not Path(RVC_MODELS_DIR / model_name).is_dir():
            rprint(f"Downloading {model_name}...")
            try:
                download_model(model_url, model_name)
            except Exception as e:
                rprint(f"Failed to download {model_name}: {e}")


def initialize() -> None:
    """Initialize the Ultimate RVC project."""
    download_base_models()
    download_sample_models()
    initialize_audio_separator()
    rprint("Initialization complete.")


if __name__ == "__main__":
    initialize()
