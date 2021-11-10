import os
import json
from absl import logging
from typing import Dict, List


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


def split_list(given_list: List, chunk_size: int) -> List:
    return [
        given_list[offs : offs + chunk_size]
        for offs in range(0, len(given_list), chunk_size)
    ]