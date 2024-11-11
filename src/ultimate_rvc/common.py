"""Common variables used in the Ultimate RVC project."""

from pathlib import Path

BASE_DIR = Path.cwd()
MODELS_DIR = BASE_DIR / "models"
RVC_MODELS_DIR = MODELS_DIR / "rvc"
SEPARATOR_MODELS_DIR = MODELS_DIR / "audio_separator"
AUDIO_DIR = BASE_DIR / "audio"
TEMP_DIR = BASE_DIR / "temp"
