import PIL.Image
import torch
import PIL
from typing import Optional, Union, List
from diffusers import AutoencoderKL
from diffusers.image_processor import VaeImageProcessor
import numpy as np

PipelineImageInput = Union[
    PIL.Image.Image,
    np.ndarray,
    torch.Tensor,
    List[PIL.Image.Image],
    List[np.ndarray],
    List[torch.Tensor],
]


class VaeProcessor():
    def __init__(self, 
                 device: Union[str, torch.device] = "cuda",
                 vae: Optional[AutoencoderKL] = None):
        super().__init__()
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        
        if vae is None:
            vae = AutoencoderKL.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", subfolder="vae")
        
        self.vae = vae.to(self.device)  # Move VAE to the specified device
        self.image_processor = VaeImageProcessor()
    
    
    def encode(self, image: torch.Tensor) -> torch.Tensor:
        image = image.to(self.device)
        with torch.no_grad():
            latents = self.vae.encode(image).latent_dist.sample()
            latents = latents*self.vae.config.scaling_factor
        return latents
    
    def encode_image(self, 
                     image: PipelineImageInput,
                     
                     ) -> torch.Tensor:
        # Preprocess image
        image = self.image_processor.preprocess(image)
        
        # Encode the image
        latents = self.encode(image)
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
        return self.image_processor.numpy_to_pil(image)