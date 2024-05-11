import os
import json
import hashlib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MDXNET_MODELS_DIR = os.path.join(BASE_DIR, "mdxnet_models")
RVC_MODELS_DIR = os.path.join(BASE_DIR, "rvc_models")
SONGS_DIR = os.path.join(BASE_DIR, "songs")


def display_progress(message, percent, progress=None):
    if progress is None:
        print(message)
    else:
        progress(percent, desc=message)


def json_dumps(thing):
    return json.dumps(
        thing,
        ensure_ascii=False,
        sort_keys=True,
        indent=4,
        separators=(",", ": "),
    )


def json_dump(thing, path):
    with open(path, "w", encoding="utf-8") as file:
        return json.dump(
            thing,
            file,
            ensure_ascii=False,
            sort_keys=True,
            indent=4,
            separators=(",", ": "),
        )


def get_hash(thing, size=5):
    return hashlib.blake2b(
        json_dumps(thing).encode("utf-8"), digest_size=size
    ).hexdigest()


def get_file_hash(filepath, size=5):
    with open(filepath, "rb") as f:
        file_hash = hashlib.blake2b(digest_size=size)
        while chunk := f.read(8192):
            file_hash.update(chunk)

    return file_hash.hexdigest()


def get_rvc_model(voice_model):
    rvc_model_filename, rvc_index_filename = None, None
    model_dir = os.path.join(RVC_MODELS_DIR, voice_model)
    for file in os.listdir(model_dir):
        ext = os.path.splitext(file)[1]
        if ext == ".pth":
            rvc_model_filename = file
        if ext == ".index":
            rvc_index_filename = file

    if rvc_model_filename is None:
        error_msg = f"No model file exists in {model_dir}."
        raise Exception(error_msg)

    return os.path.join(model_dir, rvc_model_filename), (
        os.path.join(model_dir, rvc_index_filename) if rvc_index_filename else ""
    )
