import os
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import copy


class NIHDataset(Dataset):
    """NIH Chest X-ray Dataset loader"""
    
    def __init__(self, root_dir: str, img_size: int = 224):
        """
        Args:
            root_dir: Dataset root directory
            img_size: Target image size (default: 224)
        """
        self.root = Path(root_dir)
        self.img_size = img_size
        
        if not self.root.exists():
            raise FileNotFoundError(f"Root directory not found: {self.root}")
            
        self.df = self._load_metadata()
        self._map_image_paths()
    
    def _load_metadata(self) -> pd.DataFrame:
        """Load and combine all parquet files from elixrb directory"""
        meta_dir = self.root / 'elixrb'
        if not meta_dir.exists():
            raise FileNotFoundError(f"Metadata directory not found: {meta_dir}")
            
        # Combine all parquet files
        parquet_files = list(meta_dir.glob('*.parquet'))
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in {meta_dir}")
            
        return pd.concat([pd.read_parquet(f) for f in parquet_files])
    
    def _map_image_paths(self):
        """Create mapping of image names to their full paths"""
        # Find appropriate image directory based on size
        if self.img_size:
            img_dir = self.root / f"{self.img_size}x{self.img_size}"
            if not img_dir.exists():
                img_dir = self.root / 'original'
        else:
            img_dir = self.root / 'original'
            
        if not img_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {img_dir}")
        
        # Get image paths dictionary
        paths = scan_sub_dirs(img_dir)
        
        # Create mapping from image filename to full path
        image_paths = {}
        for path_obj, full_path in paths.items():
            image_paths[path_obj.name] = full_path
        
        # Map image indices to their paths
        self.df['Image Path'] = self.df['Image Index'].map(lambda x: image_paths.get(x))
        
        # Verify all images were found
        missing_images = self.df[self.df['Image Path'].isna()]['Image Index'].tolist()
        if missing_images:
            raise FileNotFoundError(f"Could not find paths for images: {missing_images[:5]}...")

    def filter_by_label(self, label: str, exclude: bool = False) -> 'NIHDataset':
        """
        Filter dataset by label.
        Args:
            label: Target label
            exclude: If True, exclude samples with label instead of keeping them
        Returns:
            New filtered dataset instance
        """
        new_dataset = copy.deepcopy(self)
        mask = new_dataset.df['Finding Labels'].apply(lambda x: label in x)
        new_dataset.df = new_dataset.df[~mask if exclude else mask]
        return new_dataset
    
    def limit_samples(self, label: str, max_samples: int) -> 'NIHDataset':
        """
        Limit number of samples for a given label while preserving other samples.
        Args:
            label: Target label to limit
            max_samples: Maximum number of samples to keep for the target label
        Returns:
            New dataset instance with limited samples for target label
        """
        new_dataset = copy.deepcopy(self)
        has_label = new_dataset.df['Finding Labels'].apply(lambda x: label in x)
        label_df = new_dataset.df[has_label]
        other_df = new_dataset.df[~has_label]
        
        if len(label_df) > max_samples:
            label_df = label_df.sample(n=max_samples, random_state=42)
        
        new_dataset.df = pd.concat([label_df, other_df])
        return new_dataset
    
    def get_label_counts(self) -> dict:
        """Get counts of unique labels in the dataset"""
        counts = {}
        for labels in self.df['Finding Labels']:
            for label in labels:
                counts[label] = counts.get(label, 0) + 1
        return counts
    
    def select_columns(self, columns: list) -> 'NIHDataset':
        """Create new dataset with only selected columns"""
        new_dataset = copy.deepcopy(self)
        new_dataset.df = new_dataset.df[columns]
        return new_dataset
    
    def get_fields(self) -> list:
        """Get list of fields in the dataset"""
        return self.df.columns.tolist()

    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> tuple:
        return self.df.iloc[idx]




        
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from functools import partial
import os
from typing import Union

def _process_subdir(supported_formats, subdir):
    """Helper function to process each subdirectory"""
    paths_dict = {}
    
    # Ensure subdir is Path object
    subdir = Path(subdir)
    img_subdir = subdir / 'images'
    
    if not img_subdir.exists():
        return paths_dict
    
    files = [f for f in img_subdir.glob('*') if f.suffix in supported_formats]
    for f in files:
        abs_path = f.resolve()
        paths_dict[f] = abs_path
    return paths_dict


def scan_sub_dirs(img_dir: Union[str, Path]) -> dict:
   
    # Convert to Path if string
    img_dir = Path(img_dir)
    
    # Validate input
    if not img_dir.exists():
        raise FileNotFoundError(f"Directory not found: {img_dir}")
    if not img_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {img_dir}")
    
    supported_formats = ['.png', '.jpg', '.jpeg']
    paths = {}
    
    # Get subdirectories using Path
    subdirs = [d for d in img_dir.iterdir() if d.is_dir()]

    with ThreadPoolExecutor() as executor:
        process_func = partial(_process_subdir, supported_formats)
        results = executor.map(process_func, subdirs)
        
    for result in results:
        paths.update(result)
        
    return paths