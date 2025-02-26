import os
import numpy as np
from huggingface_hub import snapshot_download
from PIL import Image
from typing import Union, List, Optional

import tensorflow as tf
import tensorflow_text  # This registers the SentencepieceOp
import tensorflow_hub as tf_hub

from PIL import Image


class ELIXR:
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
        self.model_dir = model_dir or os.environ.get('HF_HOME')
        self.model_repo_id = model_repo_id
        self.verbose = verbose
        
        # Initialize models
        self.elixrc = None
        self.elixrc_infer_fn = None
        self.elixrb = None
        self.elixrb_infer_fn = None
        
        
        
        # Create model directory if it doesn't exist
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Load or download the model
        elixrc_path = os.path.join(self.model_dir, 'elixr-c-v2-pooled')
        elixrb_path = os.path.join(self.model_dir, 'pax-elixr-b-text')
        
        if download_if_missing:
            # ELIXR-C model için kontrol
            if not os.path.isdir(elixrc_path) or not os.listdir(elixrc_path):
                print(f"Downloading ELIXR-C model to {elixrc_path}") if self.verbose else None
                snapshot_download(
                    self.model_repo_id,
                    revision="main",
                    allow_patterns=["elixr-c-v2-pooled/**"],  # Alt dizinleri dahil etmek için /** kullanın
                    local_dir=self.model_dir
                )

            # ELIXR-B model için kontrol
            if not os.path.isdir(elixrb_path) or not os.listdir(elixrb_path):
                print(f"Downloading ELIXR-B model to {elixrb_path}") if self.verbose else None
                snapshot_download(
                    self.model_repo_id,
                    revision="main",
                    allow_patterns=["pax-elixr-b-text/**"],  # Alt dizinleri dahil etmek için /** kullanın
                    local_dir=self.model_dir
                )

            
        # Load the models
        self.elixrc =tf.saved_model.load(elixrc_path)
        self.elixrc_infer_fn = self.elixrc.signatures['serving_default']

        self.elixrb = tf.saved_model.load(elixrb_path)
        self.elixrb_infer_fn = self.elixrb.signatures['serving_default']

        #load bert tokenizer
        self.tokenizer = tf_hub.load("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")


    def _preprocess_text(self, text:str) -> tf.Tensor:

        out = self.tokenizer(tf.constant([text.lower()]))

        ids = out['input_word_ids'].numpy().astype(np.int32)
        masks = out['input_mask'].numpy().astype(np.float32)

        paddings = 1.0 - masks

        end_token_idx = ids ==102
        ids[end_token_idx] = 0
        paddings[end_token_idx] = 1.0
        ids = np.expand_dims(ids, axis=1)
        paddings = np.expand_dims(paddings, axis=1)

        assert ids.shape == (1, 1, 128)
        assert paddings.shape == (1, 1, 128)
        return ids, paddings


    def _preprocess_image(self, img:Image) -> np.ndarray:
        """Görüntüyü ön işler ve bir numpy dizisi döndürür."""
        if not isinstance(img, Image.Image):
            raise ValueError("Input must be a PIL Image")

        if img.mode != 'L':
            img = img.convert('L')

        # Resize image to 256x256
        if img.size != (256, 256):
            img = img.resize((256, 256))

        # Convert image to numpy array
        img_array = np.array(img)
        return img_array

    def _image_to_tfexample(self, img_array: np.ndarray) -> tf.train.Example:

        if not isinstance(img_array, np.ndarray):
            raise ValueError("Input must be a numpy array")

        # Ensure the array is 2-D (grayscale image)
        if img_array.ndim != 2:
            raise ValueError(f'Array must be 2-D. Actual dimensions: {img_array.ndim}')

        # Convert to float32 and normalize
        image = img_array.astype(np.float32)
        image -= image.min()

        if img_array.dtype == np.uint8:
            # For uint8 images, no rescaling is needed
            pixel_array = image.astype(np.uint8)
            # Convert 2D → 3D for TensorFlow
            img_array_3d = pixel_array[..., np.newaxis]
        else:
            # For other data types, scale image to use the full uint8 range
            max_val = image.max()
            if max_val > 0:
                image = image * 255.0 / max_val
            pixel_array = image.astype(np.uint8)
            # Convert 2D → 3D for TensorFlow
            img_array_3d = pixel_array[..., np.newaxis]

        encoded_img = tf.io.encode_png(img_array_3d).numpy()

        # Create TensorFlow Example
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    'image/encoded': tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[encoded_img])
                    ),
                    'image/format': tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[b'png'])
                    )
                }
            )
        )
        return example.SerializeToString()


    def __call__(self,
                    image:Image.Image,
                    text:str =None) -> tf.Tensor:
        
            
        is_text_valid = True if text else False


        if is_text_valid:
            text_ids, text_paddings = self._preprocess_text(text)
        else:
            text_ids = np.zeros((1, 1, 128), dtype=np.int32)
            text_paddings = np.ones((1, 1, 128), dtype=np.float32)

        
        preprocessed_img = self._preprocess_image(image) 

        example = self._image_to_tfexample(preprocessed_img)

        # Run inference elixr-c
        elixrc_output = self.elixrc_infer_fn(input_example = tf.constant([example]))
        elixrc_embedding = elixrc_output['feature_maps_0'].numpy()


        # Run inference elixr-b

        qformer_input = {
            'image_feature': elixrc_embedding.tolist(),
            'ids': text_ids.tolist(),
            'paddings': text_paddings.tolist()
        }

        elixrb_output = self.elixrb_infer_fn(**qformer_input)

        general_img_embedding = elixrb_output['img_emb'].numpy()[0]
        contrastive_img_embedding = elixrb_output['all_contrastive_img_emb'].numpy()[0]


        return {
            'image_embedding': elixrc_embedding,
            'general_img_embedding': general_img_embedding,
            'qformer_embedding': contrastive_img_embedding
        }


