import os
from PIL import Image
import tensorflow as tf
import numpy as np
from typing import Union, List, Optional

from ..models.elixr import ELIXRC



class ElIXRCProcessor():
    def __init__(
            self,
            device_id:Optional[int] = 0,
            model: Optional[ELIXRC] = None
    ):

        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id) 

        self.device_id = device_id
        
        if model is None:
            self.model = ELIXRC()

    

    def preprocess_image(self, img:Image) -> tf.Tensor:
        
        if not isinstance(img, Image.Image):
            raise ValueError("Input must be a PIL Image")
        
        if img.mode != 'L':
            img = img.convert('L')
        
        # Resize image to 256x256
        if img.size != (256, 256):
            img = img.resize((256, 256))
    
        # Convert image to numpy array
        img = np.array(img)

        return img
    
    def image_to_tfexample(self, img_array: np.ndarray) -> tf.train.Example:
        # Make sure the image is 2-dimensional
        if img_array.ndim != 2:
            raise ValueError(f'Array must be 2-D. Actual dimensions: {img_array.ndim}')

        # Convert 2D → 3D (format expected by TensorFlow)
        img_array_3d = img_array[..., np.newaxis]

        # Encode to PNG using TensorFlow
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

    def prepare_input(self, images:Union[List[Image.Image], Image.Image]) -> tf.Tensor:
        
        if not isinstance(images, list):
            images = [images]
        
        processed_images = [self.preprocess_image(img) for img in images]
        tf_examples = [self.image_to_tfexample(img) for img in processed_images]
        
        return tf.constant(tf_examples)
    


    def __call__(self, images:Union[List[Image.Image], Image.Image]) -> tf.Tensor:

        
        inputs = self.prepare_input(images)

        outputs = self.model.infer_fn(inputs)


        seq_embeddings = outputs['sequence_output'].numpy()
        pooled_embeddings = outputs['pooled_output'].numpy()

        return { 'sequence_output': seq_embeddings, 'pooled_output': pooled_embeddings }
    
