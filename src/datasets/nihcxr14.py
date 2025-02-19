import os
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import copy
import h5py


class NIHDataset(Dataset):
    """NIH Chest X-ray Dataset loader"""
    
    def __init__(self, root_dir: str, img_size: int = 224):
        self.root = Path(root_dir)
        self.img_size = img_size
        
        if not self.root.exists():
            raise FileNotFoundError(f"Root directory not found: {self.root}")
            
        # Load metadata and embeddings separately
        self.df, self.embeddings = self._load_metadata()
        self._map_image_paths()
    
    def _load_metadata(self) -> tuple[pd.DataFrame, np.ndarray]:
        """Load metadata and embeddings separately"""
        meta_dir = self.root / 'elixrb'
        if not meta_dir.exists():
            raise FileNotFoundError(f"Metadata directory not found: {meta_dir}")
            
        parquet_files = list(meta_dir.glob('*.parquet'))
        h5_files = list(meta_dir.glob('*.h5'))
        
        if not parquet_files or not h5_files:
            raise FileNotFoundError("Missing required files")

        dataframes = []
        embeds = []
        
        for h5_file in h5_files:
            filename = h5_file.stem
            common_name = '_'.join(filename.split('_')[:-1])
            parquet_file = meta_dir / f"{common_name}.parquet"
            
            if not parquet_file.exists():
                raise FileNotFoundError(f"Missing parquet file for {h5_file}")
            
            with h5py.File(h5_file, 'r') as f:
                embeddings = f['embeddings'][:]
                df = pd.read_parquet(parquet_file)
                
                if len(embeddings) != len(df):
                    raise ValueError(f"Mismatch in {h5_file.stem}")
                
                embeds.append(embeddings)
                dataframes.append(df)
            
        return pd.concat(dataframes), np.concatenate(embeds)

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

    def filter_by_labels(self, target_labels: list, mode: str = 'include') -> 'NIHDataset':
        """
        Filter dataset by labels while keeping embeddings synchronized.
        Only keeps samples that ONLY have the target labels.
        """
        new_dataset = copy.deepcopy(self)
        print(f"\nFiltering for labels: {target_labels}, mode={mode}")
        print("DataFrame shape before filtering:", new_dataset.df.shape)

        def check_labels(labels):
            if isinstance(labels, (list, np.ndarray)):
                # Sadece hedef etiketlere sahip örnekleri seç
                return all(label in target_labels for label in labels)
            else:
                labels_list = str(labels).split('|')
                return all(label in target_labels for label in labels_list)

        # Get boolean mask
        mask = new_dataset.df['Finding Labels'].apply(check_labels)
        selected_indices = np.where(mask)[0]

        # Filter both DataFrame and embeddings
        new_dataset.df = new_dataset.df.iloc[selected_indices]
        new_dataset.embeddings = new_dataset.embeddings[selected_indices]

        return new_dataset
    
    def limit_samples(self, label: str, max_samples: int) -> 'NIHDataset':
        """Limit samples while keeping embeddings synchronized"""
        new_dataset = copy.deepcopy(self)
        has_label = new_dataset.df['Finding Labels'].apply(lambda x: label in x)
        
        label_indices = np.where(has_label)[0]
        other_indices = np.where(~has_label)[0]
        
        if len(label_indices) > max_samples:
            selected_label_indices = np.random.choice(
                label_indices, max_samples, replace=False
            )
            selected_indices = np.concatenate([selected_label_indices, other_indices])
            
            new_dataset.df = new_dataset.df.iloc[selected_indices]
            new_dataset.embeddings = new_dataset.embeddings[selected_indices]
        
        return new_dataset
    
    def __getitem__(self, idx: int) -> tuple[pd.Series, np.ndarray]:
        """Return both metadata and embedding for an index"""
        if self.embeddings is None:
            return self.df.iloc[idx]   
        return self.df.iloc[idx], self.embeddings[idx]
    
    def __len__(self) -> int:
        return len(self.df)
    
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
    
        df_columns = [col for col in columns if col != 'Embeddings']
        keep_embeddings = 'Embeddings' in columns
    
        for col in df_columns:
            if col not in self.df.columns:
                raise ValueError(f"Column not found in DataFrame: {col}")
    
        new_dataset.df = new_dataset.df[df_columns]
    
        
        if not keep_embeddings:
            new_dataset.embeddings = None
        
        return new_dataset

    def get_fields(self) -> list:
        """Get list of fields in the dataset"""
        fields = list(self.df.columns)
        if self.embeddings is not None:
            fields = ['Embeddings'] + fields
        return fields
    
   
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