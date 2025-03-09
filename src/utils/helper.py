
import os
import json
import numpy as np
from typing import List, Union
from dataclasses import dataclass

from huggingface_hub import snapshot_download, login

# copied from https://huggingface.co/datasets/Yasintuncer/nih-cxr14-elixr/blob/main/utils/helper.py
def decode_str(data: Union[bytes, np.ndarray, List[bytes], np.bytes_]) -> List[str]:
    if isinstance(data, np.ndarray):
        if data.dtype.char == 'S' or data[0].dtype == np.bytes_:
            return list(data.astype(str))

    if isinstance(data, bytes) or isinstance(data, np.bytes_):
        return [data.decode('utf-8')]

    if isinstance(data, list):
        if all(isinstance(i, bytes) or isinstance(i, np.bytes_) for i in data):
            return [i.decode('utf-8') for i in data]

    raise ValueError('Invalid input data type')


def download(root:str, token:str)-> bool:
    try:
        if not token:
            raise ValueError('Token is required')
        login(token=token)
    except Exception as e:
        print(f"Authentication failed: {e}")
        return False

    try:
        snapshot_download(
            repo_id="Yasintuncer/nih-cxr14-elixr",
            local_dir=root,
            repo_type="dataset",
            revision="main",
            ignore_patterns=[".*"],
        )
        return True
    except HFHubHTTPError as e:
        print(f"Download failed: {e}")
        return False


@dataclass
class Feature:
    name:str
    dtype: Union[str, List[str]]
    shape: Union[list, tuple]

    def dim(self):
        return tuple(self.shape) if isinstance(self.shape, list) else (self.shape,)


@dataclass
class DsInfo:
    name:str
    features: List[Feature]
    parent: Union["DsInfo", None]

    def path(self)-> str:
        return f"{self.parent.path()}/{self.name}" if self.parent else self.name
    def ds_name(self)-> str:
        return self.parent.name + "_" + self.name if self.parent else self.name


def load_info(path:str) -> (dict, List[DsInfo]):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    
    try:
        with open(path, 'r') as f:
            info = json.load(f)
    except Exception as e:
        raise ValueError(f"Error reading file: {e}")


    data = info['info']

    def extract(data, parent=None, dsets=None)-> List[DsInfo]:
        if dsets is None:
            dsets = []

        for k,v in data.items():
            if isinstance(v, dict):
                if "features" in v and v.get("type") == "dataset":
                    name = k
                    features = [Feature(name=f_name, dtype=f_info["dtype"], shape=f_info["shape"]) for f_name, f_info in v["features"].items()]
                    dsets.append(DsInfo(name=name, features=features, parent=parent))

                # recusion for nested datasets
                extract(v, parent=DsInfo(name=k, features=None, parent=parent), dsets=dsets)
        
        return dsets

    dsets = extract(info["datasets"])
    data["existing_datasets"] = [ds.ds_name() for ds in dsets]

    return data, dsets

