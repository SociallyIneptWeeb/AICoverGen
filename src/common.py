import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
MDXNET_MODELS_DIR = os.path.join(MODELS_DIR, "mdxnet")
RVC_MODELS_DIR = os.path.join(MODELS_DIR, "rvc")
GRADIO_TEMP_DIR = os.path.join(BASE_DIR, "gradio_temp")
