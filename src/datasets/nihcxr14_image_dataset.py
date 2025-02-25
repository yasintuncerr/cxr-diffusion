import os
import pickle
from pathlib import Path
from typing import Union, Tuple, Optional

from PIL import Image
import torch
from torch.utils.data import Dataset

class NIHImageDataset(Dataset):
    """
    Dataset class for NIH CXR-14 chest X-ray images.
    
    The dataset is organized in directories images_001 through images_012,
    each containing an 'images' subdirectory with the actual image files.
    
    Args:
        root_dir (str): Root directory of the dataset
        img_size (Optional[int]): If provided, looks for resized images in {img_size}x{img_size} subdirectory
        transform (Optional[callable]): Optional transform to be applied to images
        cache_mapping (bool): Whether to cache the image path mapping to disk
    """
    def __init__(
        self, 
        root_dir: str, 
        img_size: Optional[int] = None,
        transform: Optional[callable] = None,
        cache_mapping: bool = True
    ):
        self.root = Path(root_dir)
        if not self.root.exists():
            raise FileNotFoundError(f"Root directory not found: {self.root}")
        
        self.img_size = img_size
        self.transform = transform
        self._image_mapping = {}
        
        self._scan_nih_cxr14_directory()
        
        if cache_mapping and self._check_cache_mapping():
            return
            
        self._scan_files()

    def get_image_ids(self) -> Tuple[str]:
        """Get all image IDs in the dataset."""
        return tuple(self._image_mapping.keys())


    def _check_cache_mapping(self) -> bool:
        """Check if the image mapping is already cached and load it if available."""
        cache_file = self.root / '.nih_cxr14_image_id_path_mapping.pkl'
        try:
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    self._image_mapping = pickle.load(f)
                return True
        except (pickle.PickleError, IOError) as e:
            print(f"Warning: Failed to load cache mapping: {e}")
        return False

    def _scan_nih_cxr14_directory(self) -> None:
        """Verify and set up the NIH CXR-14 directory structure."""
        if self.img_size is not None:
            resized_dir = self.root / f"{self.img_size}x{self.img_size}"
            if resized_dir.exists():
                self.root = resized_dir

        if not self.root.exists():
            raise FileNotFoundError(f"Root directory not found: {self.root}")
        
        # Check for at least one image directory
        if not any(entry.name.startswith('images_') and entry.is_dir() 
                  for entry in self.root.iterdir()):
            raise FileNotFoundError(
                f"NIH CXR-14 directory structure not found in {self.root}"
            )

    def _scan_files(self) -> None:
        """Scan and map all image files in the dataset directory structure."""
        # Find all image directories dynamically
        self.dirs = sorted([
            d for d in self.root.iterdir() 
            if d.is_dir() and d.name.startswith('images_')
        ])
        
        for directory in self.dirs:
            images_dir = directory / 'images'
            if not images_dir.exists():
                continue  # Skip directories without images subdirectory
            
            # Update mapping with all images in current directory
            self._image_mapping.update({
                img.name: img 
                for img in images_dir.iterdir() 
                if img.suffix.lower() in ['.png', '.jpg', '.jpeg']
            })

        if not self._image_mapping:
            raise FileNotFoundError(f"No valid image files found in {self.root}")

        # Cache the mapping
        try:
            cache_file = self.root / '.nih_cxr14_image_id_path_mapping.pkl'
            with open(cache_file, 'wb') as f:
                pickle.dump(self._image_mapping, f)
        except (pickle.PickleError, IOError) as e:
            print(f"Warning: Failed to cache mapping: {e}")

    def __len__(self) -> int:
        """Returns the total number of images in the dataset."""
        return len(self._image_mapping)

    def __getitem__(self, idx: Union[int, str]) -> Tuple[Union[Image.Image, torch.Tensor], str]:
        """Retrieve an image and its ID from the dataset."""
        try:
            if isinstance(idx, str):
                img_id = idx
                if img_id not in self._image_mapping:
                    raise KeyError(f"Image ID '{img_id}' not found in dataset")
            elif isinstance(idx, int):
                if idx < 0 or idx >= len(self._image_mapping):
                    raise IndexError("Index out of range")
                img_id = list(self._image_mapping.keys())[idx]
            else:
                raise ValueError("Index must be an integer or a string")
    
            img_path = self._image_mapping[img_id]
            
            try:
                img = Image.open(img_path)
                img = img.convert('RGB')
                
                if self.transform:
                    img = self.transform(img)
                
                # Make a copy to ensure we're not returning a reference that might be closed
                img_copy = img.copy()
                return img_copy, img_id
            except (IOError, OSError) as e:
                raise IOError(f"Error loading image {img_id}: {str(e)}")
            finally:
                # Close the original image, but we're returning a copy
                if hasattr(img, 'close'):
                    img.close()
                    
        except Exception as e:
            raise RuntimeError(f"Error accessing item at {idx}: {str(e)}")