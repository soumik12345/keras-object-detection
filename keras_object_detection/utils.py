import os
import json
from typing import Dict
from absl import logging


def make_directory(dir_path):
    if not os.path.exists(dir_path):
        logging.info(f"Making Directory {dir_path}")
        os.mkdir(dir_path)


def dump_dictionary_as_json(dictionary: Dict, json_file_path, indent: int = 4):
    with open(json_file_path, "w") as handle:
        json.dump(dictionary, handle, indent=indent)


def load_json_as_dict(json_file_path) -> Dict:
    with open(json_file_path, "r") as handle:
        loaded_dictionary = json.loads(handle.read())
    return loaded_dictionary
