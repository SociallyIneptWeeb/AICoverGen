"""
Package which defines modules that facilitate RVC based audio
generation.
"""

import static_ffmpeg
import static_sox

from ultimate_rvc.core.common import download_base_models

download_base_models()
static_ffmpeg.add_paths()
static_sox.add_paths()
