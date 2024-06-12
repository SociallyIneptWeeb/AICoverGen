import os
import json
import hashlib
import shutil

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MDXNET_MODELS_DIR = os.path.join(BASE_DIR, "mdxnet_models")
RVC_MODELS_DIR = os.path.join(BASE_DIR, "rvc_models")
SONGS_DIR = os.path.join(BASE_DIR, "songs")
TEMP_AUDIO_DIR = os.path.join(SONGS_DIR, "temp")


def display_progress(message, percent=None, progress=None):
    if progress is None:
        print(message)
    else:
        progress(percent, desc=message)


def remove_suffix_after(text: str, occurrence: str):
    location = text.rfind(occurrence)
    if location == -1:
        return text
    else:
        return text[: location + len(occurrence)]


def copy_files_to_new_folder(file_paths, folder_path):
    os.makedirs(folder_path)
    for file_path in file_paths:
        if not os.path.exists(file_path):
            raise Exception(f"File not found: {file_path}")
        shutil.copyfile(
            file_path, os.path.join(folder_path, os.path.basename(file_path))
        )


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


def json_load(path, encoding=None):
    with open(path, encoding=encoding) as file:
        return json.load(file)


def get_hash(thing, size=5):
    return hashlib.blake2b(
        json_dumps(thing).encode("utf-8"), digest_size=size
    ).hexdigest()


# TODO consider increasing size to 16
# otherwise we might have problems with hash collisions
# when using app as CLI
def get_file_hash(filepath, size=5):
    with open(filepath, "rb") as f:
        file_hash = hashlib.blake2b(digest_size=size)
        while chunk := f.read(8192):
            file_hash.update(chunk)

    return file_hash.hexdigest()


def get_rvc_model(voice_model):
    rvc_model_filename, rvc_index_filename = None, None
    model_dir = os.path.join(RVC_MODELS_DIR, voice_model)
    if not os.path.exists(model_dir):
        raise Exception(f"Voice model directory '{voice_model}' does not exist.")
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
