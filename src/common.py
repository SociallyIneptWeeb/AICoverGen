import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MDXNET_MODELS_DIR = os.path.join(BASE_DIR, "mdxnet_models")
RVC_MODELS_DIR = os.path.join(BASE_DIR, "rvc_models")
GRADIO_TEMP_DIR = os.path.join(BASE_DIR, "gradio_temp")
