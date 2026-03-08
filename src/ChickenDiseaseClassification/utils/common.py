import os
import yaml
import json
import joblib
from box import ConfigBox
from pathlib import Path
from typing import Any
from ensure import ensure_annotations
from box.exceptions import BoxValueError


@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """Reads a yaml file and returns a ConfigBox object.

    Args:
        path_to_yaml (Path): path to the yaml file

    Raises:
        ValueError: if yaml file is empty
        e: empty file

    Returns:
        ConfigBox: ConfigBox type
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            if content is None:
                raise ValueError("yaml file is empty")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e


@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """Create a list of directories.

    Args:
        path_to_directories (list): list of paths of directories
        verbose (bool, optional): log creation. Defaults to True.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            print(f"Created directory at: {path}")


@ensure_annotations
def get_size(path: Path) -> str:
    """Get size of a file in KB.

    Args:
        path (Path): path of the file

    Returns:
        str: size in KB
    """
    size_in_kb = round(os.path.getsize(path) / 1024)
    return f"~ {size_in_kb} KB"


@ensure_annotations
def save_json(path: Path, data: dict):
    """Save data to a JSON file.

    Args:
        path (Path): path to the json file
        data (dict): data to be saved in json file
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    print(f"JSON file saved at: {path}")


import base64

def decodeImage(imgstring, fileName):
    """Decode a base64 image string and save it to a file.

    Args:
        imgstring (str): base64 encoded image string
        fileName (str): path to save the decoded image
    """
    imgdata = base64.b64decode(imgstring)
    with open(fileName, 'wb') as f:
        f.write(imgdata)
