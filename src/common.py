import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
RVC_MODELS_DIR = os.path.join(MODELS_DIR, "rvc")
SEPARATOR_MODELS_DIR = os.path.join(MODELS_DIR, "audio_separator")
AUDIO_DIR = os.path.join(BASE_DIR, "audio")
GRADIO_TEMP_DIR = os.path.join(AUDIO_DIR, "gradio_temp")
