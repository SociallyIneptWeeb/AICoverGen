from pathlib import Path
import requests

# Download links for each model type
RVC_other_DOWNLOAD_LINK = 'https://huggingface.co/Politrees/RVC_resources/resolve/main/predictors/'
RVC_hubert_DOWNLOAD_LINK = 'https://huggingface.co/Politrees/RVC_resources/resolve/main/embedders/'
MDX_DOWNLOAD_LINK = 'https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/'
RVC_DOWNLOAD_LINK = 'https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/'

# Base directory for saving models
BASE_DIR = Path(__file__).resolve().parent.parent
rvc_models_dir = BASE_DIR / 'rvc_models'
mdxnet_models_dir = BASE_DIR / 'mdxnet_models'

# Function to download models
def dl_model(link, model_name, dir_name):
    with requests.get(f'{link}{model_name}') as r:
        r.raise_for_status()
        with open(dir_name / model_name, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

# Main download logic
if __name__ == '__main__':
    # RVC models
    rvc_other_names = ['rmvpe.pt', 'fcpe.pt']
    for model in rvc_other_names:
        print(f'Downloading {model}...')
        dl_model(RVC_other_DOWNLOAD_LINK, model, rvc_models_dir)

    rvc_hubert_names = ['hubert_base.pt']
    for model in rvc_hubert_names:
        print(f'Downloading {model}...')
        dl_model(RVC_hubert_DOWNLOAD_LINK, model, rvc_models_dir)

    # MDX models
    mdx_model_names = ['UVR-MDX-NET-Voc_FT.onnx', 'UVR_MDXNET_KARA_2.onnx', 'Reverb_HQ_By_FoxJoy.onnx']
    for model in mdx_model_names:
        print(f'Downloading {model}...')
        dl_model(MDX_DOWNLOAD_LINK, model, mdxnet_models_dir)

    print('All models downloaded!')
