import os
import pandas as pd
from PIL import Image
from typing import List, Union, Optional
import copy


class NIHImageDataset():
    def __init__(
            self,
            root: str,
            transform: Optional[callable] = None,
            image_mode = "RGB",
            use_cache: bool = False
    ):
        self.root = root
        self.transform = transform
        self.use_cache = use_cache
        self.image_mode = image_mode
        self.data_frame = None

        
        dirs = self._validate()
        if len(dirs) ==0:
            raise ValueError("No subdirectories found in the root directory")
        
        self._initialize_data(dirs)


    def _validate(self):
        if not os.path.exists(self.root):
            raise FileNotFoundError(f"Directory not found: {self.root}")
        if not os.path.isdir(self.root):
            raise ValueError("Invalid directory path")
        
        # get dirs 
        dirs = [d for d in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, d))]
        if not dirs:
            raise ValueError("No subdirectories found in the root directory")
        
        # dirs should contain images directory
        for d in dirs:
            path = os.path.join(self.root, d)
            subdirs = [sd for sd in os.listdir(path) if os.path.isdir(os.path.join(path, sd))]
            if "images" not in subdirs:
                raise ValueError(f"Directory {d} does not contain 'images' subdirectory")
        
        return dirs
    
    def _initialize_data(self, dirs):
        dataframes = []
        for d in dirs:
            path = os.path.join(self.root, d, "images")
            if self.use_cache:
                df = self._load_from_cache(path)
                if df is not None:
                    dataframes.append(df)
                    continue
            df = self._create_data_frame(path)
            if self.use_cache:
                self._save_to_cache(df, path)
            dataframes.append(df)
        self.data_frame = pd.concat(dataframes)

    def _create_data_frame(self, path):
        data = {"Image Index": [], "Path": []}
        for root, _, files in os.walk(path):
            for f in files:
                data["Image Index"].append(f)
                data["Path"].append(os.path.join(root, f))
        return pd.DataFrame(data)
    
    def _load_from_cache(self, path):
        cache_path = os.path.join(path, ".cache.csv")
        if not os.path.exists(cache_path):
            return None
        return pd.read_csv(cache_path)
    
    def _save_to_cache(self, df, path):
        cache_path = os.path.join(path, ".cache.csv")
        df.to_csv(cache_path, index=False)

    
    def __len__(self):
        return len(self.data_frame)
    
    def filter_by_id(self, ids: List[str]):
        new_instance = copy.deepcopy(self)
        new_instance.data_frame = new_instance.data_frame[new_instance.data_frame["Image Index"].isin(ids)]
        return new_instance
    
    def to_dict(self):
        return self.data_frame.to_dict(orient="records")
    
    def __getitem__(self, idx: Union[int, str]):
        row = None
        if isinstance(idx, int):
            row = [self.data_frame.iloc[idx]]
        elif isinstance(idx, str):
            row = self.data_frame[self.data_frame["Image Index"] == idx]
        else:
            raise ValueError("Invalid index type")
    
        if row is None or row.empty:
            raise ValueError(f"Index '{idx}' not found")
        
        image_idx = row["Image Index"].values[0]
        path = row["Path"].values[0]

        image = Image.open(path)
        if self.image_mode == "RGB":
            image = image.convert("RGB")
        elif self.image_mode == "L":
            image = image.convert("L")
        else:
            raise ValueError(f"Invalid image mode: {self.image_mode}")
        
        if self.transform:
            image = self.transform(image)

        return image, image_idx
