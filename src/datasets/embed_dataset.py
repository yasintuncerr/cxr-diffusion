import os
import h5py
import pandas as pd
from typing import List
from ..utils import decode_str
import copy
import json

## copied from https://huggingface.co/datasets/Yasintuncer/nih-cxr14-elixr/blob/main/utils/embed_dataset.py

class EmbeddingDataset:

    def __init__(self, root: str, name: str, use_cache: bool = False):
        self.root = root
        self.name = name
        self.use_cache = use_cache
        self.files = None
        self.data_frame = None
        self.offsets = None
        self.h5_cols = None
        self.active_cols = None
        self._validate_root()
        self._initialize_data()

    def _validate_root(self):
        if not os.path.exists(self.root):
            raise FileNotFoundError(f"Directory not found: {self.root}")

        if os.path.isfile(self.root):
            if not self.root.endswith(".h5"):
                raise ValueError("Invalid file format")
            self.files = [self.root]
        else:
            self.files = sorted([os.path.join(self.root, f) for f in os.listdir(self.root) if f.endswith(".h5")],
                                key=lambda x: int(os.path.splitext(x.split("_")[-1])[0]))

    def _initialize_data(self):
        if self.use_cache and self._load_from_cache():
            return
        self._create_data_from_h5()
        if self.use_cache:
            self._save_to_cache()
        if self.h5_cols:
            self.active_cols = self.h5_cols.copy()
        else:
            self.active_cols = []

    def _load_from_cache(self):
        cache_path = os.path.join(self.root, f".cache{self.name}.csv")
        if not os.path.exists(cache_path):
            return False
        with open(cache_path, 'r') as f:
            lines = f.readlines()
            metadata_line = lines[0]
            if metadata_line.startswith("#METADATA#"):
                metadata = json.loads(metadata_line[10:].strip())
                self.offsets = metadata.get("offsets")
                self.h5_cols = metadata.get("h5_cols")
                self.active_cols = self.h5_cols.copy()
            self.data_frame = pd.read_csv(cache_path, skiprows=1)
        return True

    def _create_data_from_h5(self):
        dfs = []
        self.offsets = []
        self.h5_cols = []
        for i, file_path in enumerate(self.files):
            try:
                df, offset, cols = self._process_h5_file(file_path, i)
                dfs.append(df)
                self.offsets.append(offset)
                if self.h5_cols and self.h5_cols != cols:
                    raise ValueError("Broken datasets: columns mismatch")
                self.h5_cols = cols
            except (FileNotFoundError, KeyError) as e:
                print(f"Error: {e}")
        self.data_frame = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
        print(f"h5_cols in _create_data_from_h5: {self.h5_cols}")  # Hata ayıklama

    def _process_h5_file(self, file_path, file_id):
        with h5py.File(file_path, 'r') as h5_file:
            img_index = h5_file["Image Index"][:]
            img_index_decoded = decode_str(img_index)
            orig_index = list(range(len(img_index_decoded)))
            df = pd.DataFrame({"Image Index": img_index_decoded, "original_index": orig_index, "file_id": [file_id] * len(img_index_decoded)})
            offset = len(img_index_decoded)
            cols = list(h5_file.keys())
            print(f"Cols in _process_h5_file: {cols}")  # Hata ayıklama
        return df, offset, cols

    def _save_to_cache(self):
        cache_path = os.path.join(self.root, f".cache{self.name}.csv")
        metadata = {"offsets": self.offsets, "h5_cols": self.h5_cols}
        with open(cache_path, "w") as f:
            f.write(f"#METADATA# {json.dumps(metadata)}\n")
            self.data_frame.to_csv(f, index=False)

    def __len__(self):
        return len(self.data_frame)

    def _find_row_info(self, index: int):
        if index >= len(self):
            raise IndexError("Index out of range")
        row = self.data_frame.iloc[index]
        return row["file_id"], row["original_index"]

    def _get_row(self, index: int):
        file_id, orig_index = self._find_row_info(index)
        with h5py.File(self.files[file_id], 'r') as h5_file:
            row = {}
            for col in self.active_cols:
                row[col] = h5_file[col][orig_index]
            row["Image Index"] = decode_str(h5_file["Image Index"][orig_index])
        return row

    def get_row_by_id(self, img_id: str):
        index = self.data_frame[self.data_frame["Image Index"] == img_id].index[0]
        return self._get_row(index)

    def get_rows_by_slice(self, start: int, end: int):
        if start is None or start < 0 or start >= len(self) or end is None or end < 0 or end > len(self):
            raise ValueError("Invalid slice indices")
        file_groups = {}
        for i in range(start, end):
            file_id, orig_index = self._find_row_info(i)
            file_groups.setdefault(file_id, []).append(orig_index)
        rows = {}
        for file_id, indices in file_groups.items():
            with h5py.File(self.files[file_id], 'r') as h5_file:
                row={}
                for col in self.active_cols:
                    row[col] = h5_file[col][indices[0]:indices[-1] + 1]
                row["Image Index"] = decode_str(h5_file["Image Index"][indices[0]:indices[-1] + 1])
                rows.update(row)
        return rows

    def get_dataframe(self):
        return self.data_frame

    def __getitem__(self, index):
        if isinstance(index, int):
            return self._get_row(index)
        elif isinstance(index, str):
            return self.get_row_by_id(index)
        elif isinstance(index, slice):
            return self.get_rows_by_slice(index.start, index.stop)
        else:
            raise ValueError("Invalid index type")

    def list_columns(self):
        return self.active_cols

    def set_active_columns(self, columns: List[str]):
        if not all(col in self.h5_cols for col in columns):
            raise ValueError("Invalid columns specified")
        self.active_cols = columns.copy()
        return self.active_cols

    def reset_active_columns(self):
        self.active_cols = self.h5_cols.copy()
        return self.active_cols

    def filter_by_ids(self, ids: List[str]):
        new_instance = copy.deepcopy(self)
        new_instance.data_frame = self.data_frame[self.data_frame["Image Index"].isin(ids)]
        file_groups = new_instance.data_frame.groupby("file_id")
        new_instance.offsets = [len(group) for _, group in file_groups]
        return new_instance