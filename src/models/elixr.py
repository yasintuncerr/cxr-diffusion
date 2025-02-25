import os
import numpy as np
import tensorflow as tf
from huggingface_hub import snapshot_download
from PIL import Image
from typing import Union, List, Optional
import tensorflow_text  # This registers the SentencepieceOp


class ELIXRC:
    def __init__(self, 
                 model_dir: Optional[str] = None,
                 download_if_missing: bool = True,
                 model_repo_id: str = "google/cxr-foundation",
                 verbose: bool = True):
        """
        Initialize the ELIXR model.
        
        Args:
            model_dir: Directory to save/load the model from
            download_if_missing: Whether to download the model if not found
            model_repo_id: HuggingFace repo ID for the model
            verbose: Whether to print verbose logging messages
        """
        self.model_dir = model_dir or '/tmp/elixr-model'
        self.model_repo_id = model_repo_id
        self.verbose = verbose
        
        # Initialize model
        self.model = None
        self.infer_fn = None
        
        # Create model directory if it doesn't exist
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Load or download the model
        elixrc_path = os.path.join(self.model_dir, 'elixr-c-v2-pooled')
        
        if download_if_missing and not os.path.exists(os.path.join(elixrc_path, 'saved_model.pb')):
            if self.verbose:
                print(f"ELIXR model not found in {elixrc_path}. Downloading...")
            snapshot_download(
                repo_id=self.model_repo_id,
                local_dir=self.model_dir,
                allow_patterns=['elixr-c-v2-pooled/*']
            )
            if self.verbose:
                print(f"ELIXR model downloaded to {elixrc_path}")
        
        # Load the model
        self.load_model()
    
    def load_model(self):
        """Load ELIXR model."""
        elixrc_path = os.path.join(self.model_dir, 'elixr-c-v2-pooled')

        if os.path.exists(os.path.join(elixrc_path, 'saved_model.pb')):
            try:
                self.model = tf.saved_model.load(elixrc_path)
                if hasattr(self.model, 'signatures') and 'serving_default' in self.model.signatures:
                    self.infer_fn = self.model.signatures['serving_default']
                    if self.verbose:
                        print(f"ELIXR model loaded successfully from {elixrc_path}")
                else:
                    raise ValueError("Model loaded but 'serving_default' signature not found")
            except Exception as e:
                print(f"Error loading ELIXR model: {e}")
                print(f"Model path: {elixrc_path}")
                print(f"Files in directory: {os.listdir(elixrc_path)}")
                raise  # Re-raise the exception for better debugging
        else:
            raise FileNotFoundError(f"ELIXR model not found at {elixrc_path}")