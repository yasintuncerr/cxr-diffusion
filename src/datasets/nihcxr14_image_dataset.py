
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
    
    The dataset is organized in directories images001 through images014,
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

    def _check_cache_mapping(self) -> bool:
        """
        Check if the image mapping is already cached and load it if available.
        
        Returns:
            bool: True if cache was loaded successfully, False otherwise
        """
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
        """
        Verify and set up the NIH CXR-14 directory structure.
        
        Raises:
            FileNotFoundError: If required directory structure is not found
        """
        if self.img_size is not None:
            resized_dir = self.root / f"{self.img_size}x{self.img_size}"
            if resized_dir.exists():
                self.root = resized_dir

        if not self.root.exists():
            raise FileNotFoundError(f"Root directory not found: {self.root}")
        
        if not any(f"images{i:03}" in os.listdir(self.root) for i in range(1, 15)):
            raise FileNotFoundError(
                f"NIH CXR-14 directory structure not found in {self.root}"
            )

    def _scan_files(self) -> None:
        """
        Scan and map all image files in the dataset directory structure.
        
        Raises:
            FileNotFoundError: If 'images' subdirectory is not found
        """
        self.dirs = [self.root / f"images{i:03}" for i in range(1, 15)]
        
        for directory in self.dirs:
            images_dir = directory / 'images'
            if not images_dir.exists():
                raise FileNotFoundError(f"Images directory not found in {directory}")
            
            # Update mapping with all images in current directory
            self._image_mapping.update({
                img.name: img 
                for img in images_dir.iterdir() 
                if img.suffix.lower() in ['.png', '.jpg', '.jpeg']
            })

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
        """
        Retrieve an image and its ID from the dataset.
        
        Args:
            idx: Either an integer index or string image ID
                
        Returns:
            tuple: (image, image_id) where image can be either PIL Image or torch.Tensor 
                   depending on whether transforms were applied
                
        Raises:
            KeyError: If image ID not found
            IndexError: If integer index out of range
            ValueError: If idx is neither int nor str
            IOError: If image file cannot be loaded
        """
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
                    img = self.transform(img)  # This might return a torch.Tensor
                    
                return img, img_id
            except (IOError, OSError) as e:
                raise IOError(f"Error loading image {img_id}: {str(e)}")
            finally:
                if hasattr(img, 'close'):
                    img.close()
                    
        except Exception as e:
            raise RuntimeError(f"Error accessing item at {idx}: {str(e)}")