import PIL.Image
import torch
import PIL
from typing import Optional, Union, List
from diffusers import AutoencoderKL
from diffusers.image_processor import VaeImageProcessor
import numpy as np


class VaeProcessor:
    def __init__(self, 
                 device: Union[str, torch.device] = "cuda",
                 vae: Optional[AutoencoderKL] = None):
        
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        
        if vae is None:
            vae = AutoencoderKL.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", subfolder="vae")
        
        self.vae = vae.to(self.device)  # Move VAE to the specified device
        self.image_processor = VaeImageProcessor()

    @staticmethod
    def numpy_to_pil(images):
        """
        Convert a NumPy image or a batch of images to a PIL image.
        """
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        if images.shape[-1] == 1:
            # special case for grayscale (single channel) images
            pil_images = [PIL.Image.fromarray(image.squeeze(), mode="L") for image in images]
        else:
            pil_images = [PIL.Image.fromarray(image) for image in images]

        return pil_images

    def encode_image(self, image: Union[PIL.Image.Image, List[PIL.Image.Image]]) -> torch.Tensor:
        if isinstance(image, (PIL.Image.Image, list)):
            processed_image = self.image_processor.preprocess(image).to(self.device)
        else:
            raise ValueError("Input must be a PIL image or a list of PIL images")
        
        # Encode images
        with torch.no_grad():
            latents = self.vae.encode(processed_image).latent_dist.sample()
            latents = self.vae.config.scaling_factor * latents

        return latents

    def decode_latent(self, latent: torch.Tensor) -> Union[PIL.Image.Image, List[PIL.Image.Image]]:
        # Scale the latents
        latents = latent / self.vae.config.scaling_factor
        
        # Make sure latents are on the correct device
        latents = latents.to(self.device)
        
        # Decode the latents
        with torch.no_grad():
            images = []
            for i in range(latents.shape[0]):
                # Decode one sample at a time to avoid CUDA out of memory
                image = self.vae.decode(latents[i:i+1]).sample
                images.append(image)
            
            # Concatenate all images
            image = torch.cat(images, dim=0)
        
        # Convert to numpy and process
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        image = np.clip(image / 2 + 0.5, 0, 1)
        
        # Convert to PIL images
        return self.numpy_to_pil(image)