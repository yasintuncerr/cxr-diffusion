import os
import copy
from pathlib import Path
from typing import List, Union
import numpy as np
from ..utils import load_info, download
from .embed_dataset import EmbeddingDataset

#copied from https://huggingface.co/datasets/Yasintuncer/nih-cxr14-elixr/blob/main/datasets/nihcxr14_elixr_dataset.py
class NIHElixrDataset:
    def __init__(self,
                 root: str = None,
                 hf_token: str = None,
                 download_missing: bool = True,
                 use_cache: bool = False):

        self.root = root
        self.hf_token = hf_token
        self.use_cache = use_cache
        if root is None:
            self.root = os.path.join(os.getenv("HF_HOME"), "datasets", "nihcxr14-elixr")

        self.root = Path(self.root).resolve()

        if not os.path.exists(self.root):
            if download_missing:
                if not download(self.root, self.hf_token):
                    raise ValueError("Error downloading dataset")
            else:
                raise ValueError("Dataset directory does not exist. Please Check or Try setting download_missing=True")

        print(f"Dataset root: {self.root}")
        self._load_datasets()
        self._validate()

    def _load_datasets(self):
        self.info, self.dsets_info = load_info(os.path.join(self.root, "datasets.json"))
        self.datasets = []

        try:
            for ds_info in self.dsets_info:
                path = os.path.join(self.root, ds_info.path())
                dset = EmbeddingDataset(path, ds_info.ds_name(), use_cache=self.use_cache)
                self.datasets.append(dset)
            self._active_datasets = list(range(len(self.datasets)))
        except Exception as e:
            raise ValueError(f"Error loading datasets: {e}")

    def _validate(self):
        idx_lists = [len(dset) for dset in self.datasets]
        if len(set(idx_lists)) > 1:
            raise ValueError("Datasets have different lengths")

        # "Image Index" ve "original_index" tutarlılığını kontrol et
        if self.datasets:
            first_dset = self.datasets[0].get_dataframe()
            for i, dset in enumerate(self.datasets[1:]):
                other_dset = dset.get_dataframe()
                if not first_dset["Image Index"].equals(other_dset["Image Index"]):
                    raise ValueError(f"Image Index mismatch between datasets 0 and {i + 1}")
                if not first_dset["original_index"].equals(other_dset["original_index"]):
                    raise ValueError(f"Original index mismatch between datasets 0 and {i + 1}")

        self.total_len = idx_lists[0] if idx_lists else 0

    def filter_by_id(self, ids: List[str]):
        new_instance = copy.deepcopy(self)
        new_instance.datasets = [dset.filter_by_ids(ids) for dset in new_instance.datasets]
        new_instance._validate()
        return new_instance

    def __len__(self):
        return self.total_len

    def deactivate_dataset(self, idx: int):
        if isinstance(idx, list):
            for i in idx:
                self.deactivate_dataset(i)
            return
        if idx in self._active_datasets:
            self._active_datasets.remove(idx)
        else:
            raise ValueError(f"Dataset {idx} is not active")

    def list_active_datasets(self):
        return [self.datasets[i].name for i in self._active_datasets]

    def activate_dataset(self, idx: int):
        if idx not in self._active_datasets:
            self._active_datasets.append(idx)
        else:
            raise ValueError(f"Dataset {idx} is already active")

    def __getitem__(self, idx: Union[int, str, slice]):
        items = {}
        try:
            for active_idx in self._active_datasets:
                d_name = self.datasets[active_idx].name
                items[d_name] = self.datasets[active_idx][idx]

            if isinstance(idx, slice) and self._active_datasets:
                ref_img_idx = list(items[self.datasets[self._active_datasets[0]].name]["Image Index"])
                for d_name, item in items.items():
                    if d_name == self.datasets[self._active_datasets[0]].name:
                        continue
                    img_idx = list (item["Image Index"])
                    if ref_img_idx != img_idx:
                        raise ValueError("Image Index mismatch between datasets")
            
        except Exception as e:
            raise ValueError(f"Error getting item: {e}")
        return items