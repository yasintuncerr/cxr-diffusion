import os
from pathlib import Path
from typing import Union, Tuple, Optional
import h5py
import numpy as np

import torch
from torch.utils.data import Dataset

from huggingface_hub import login, snapshot_download
from huggingface_hub.utils import HfHubHTTPError

import dotenv 
dotenv.load_dotenv()


def _download_embeddings_dataset(token: str, download_dir: str) -> bool:
    """
    Download dataset from Huggingface Hub repository.
    
    Args:
        token (str): Huggingface Hub authentication token
        download_dir (str): Directory path to download the dataset
        
    Returns:
        bool: True if download successful, False otherwise
    """
    # Validate token
    try:
        if token is None:
            raise ValueError("Please provide a valid token")
        login(token)
    except HfHubHTTPError as e:
        print(f"Hub authentication error: {e}")
        return False
    except Exception as e:
        print(f"Unexpected Login Error: {e}")
        return False

    # Create directory
    try:
        Path(download_dir).mkdir(parents=True, exist_ok=True)
    except PermissionError as e:
        print(f"Permission denied to create directory: {e}")
        return False
    except OSError as e:
        print(f"OS Error when creating directory: {e}")
        return False

    # Download repository
    try:
        snapshot_download(
            repo_id="8bits-ai/nih-cxr14-elixr-b-text-embeddings",
            local_dir=download_dir,
            repo_type="dataset",
            revision="main",
            ignore_patterns=[".*"],
        )
        return True
    except Exception as e:
        print(f"Error downloading repository: {e}")
        return False


class NIHElixrbDataset(Dataset):
    """
    Dataset class for NIH-CXR-14 chest X-ray ELIXRB embeddings.
    """
    def __init__(
        self,
        root_dir: Union[str, Path],
        transform: Optional[callable] = None,        
    ):
        self.root = Path(root_dir)
        if not self.root.exists():
            raise FileNotFoundError(f"Root directory not found: {self.root}")
        
        self.transform = transform
        self._initialize_dataset()

    def _initialize_dataset(self):
        """
        Initialize dataset by loading metadata and downloading if necessary.
        """
        # Download dataset if not found
        if not os.path.exists(self.root / "elixrb"):
            print("Downloading dataset...")
            if not _download_embeddings_dataset(token=os.getenv("HF_TOKEN"), download_dir=self.root):
                raise FileNotFoundError("Dataset download failed")

        self._h5_files = list((self.root / "elixrb").glob("*.h5"))

        self.h5_image_mappings = []
        self.file_sizes = []
        self.count = 0

        for h5_file in self._h5_files:
            with h5py.File(h5_file, "r") as f:
                image_ids = f["image_ids"][:]
                image_ids = self._decode_image_id(image_ids)
                belong_map = {h5_file: image_ids}

                self.h5_image_mappings.append(belong_map)
                self.file_sizes.append(len(image_ids))
                self.count += len(image_ids)

    def _decode_image_id(self, img_ids: Union[bytes, list]) -> Union[str, list]:
        """
        In elixrb we used a custom encoding for image IDs to store additional information.
        This method decodes the image ID to extract the original image ID.
        """
        if isinstance(img_ids, bytes):
            return img_ids.decode('utf-8')
        elif isinstance(img_ids, list):
            return [i.decode('utf-8') for i in img_ids]
        return img_ids

    def __len__(self) -> int:
        return sum(self.file_sizes)

    def _find_by_image_id(self, image_id: str) -> Tuple[str, int]:
        """
        Find the file containing the given image_id and its index.
        
        Args:
            image_id (str): ID of the image to find
            
        Returns:
            tuple: (filepath, index in file)
        """
        image_id = image_id.split(".")[0]

        for file_map in self.h5_image_mappings:
            for filepath, image_ids in file_map.items():
                if image_id in image_ids:
                    return filepath, np.where(image_ids == image_id)[0][0]
    
        raise ValueError(f"Image ID not found: {image_id}")

    def __getitem__(self, idx: Union[str, int]) -> Tuple[torch.Tensor, str]:
        """
        Get embedding by index or image_id.

        Args:
            idx: Either integer index or string image_id

        Returns:
            tuple: (embedding, image_id)
        """
        if isinstance(idx, str):
            filepath, index = self._find_by_image_id(idx)

        elif isinstance(idx, int):
            if idx < 0 or idx >= sum(self.file_sizes):
                raise IndexError("Index out of range")

            # Find which file contains this index
            cumsum = 0
            for i, length in enumerate(self.file_sizes):
                if cumsum + length > idx:
                    # Found the file
                    filepath = list(self.h5_image_mappings[i].keys())[0]
                    index = idx - cumsum
                    break
                cumsum += length
        else:
            raise ValueError("Index must be an integer or string")

        # Load embedding from the file
        with h5py.File(filepath, "r") as f:
            embedding = f["embeddings"][index]
            image_id = f["image_ids"][index]
            image_id = self._decode_image_id(image_id)

        if self.transform:
            embedding = self.transform(embedding)
        else:
            embedding = torch.tensor(embedding)

        return embedding, image_id