import PIL.Image
import torch
import PIL
from typing import Optional, Union, List
from diffusers import AutoencoderKL
from diffusers.image_processor import VaeImageProcessor

class VaeProcessor:
    def __init__(self, 
                 vae: AutoencoderKL = AutoencoderKL.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", subfolder="vae"),
                 device: str = "cuda"):
        
        self.device = device
        self.vae = vae.to(self.device)  # Move VAE to the specified device
        self.image_processor = VaeImageProcessor()

    def encode_image(self, image: Union[PIL.Image.Image, torch.Tensor, List[PIL.Image.Image], List[torch.Tensor]]) -> torch.Tensor:
        if isinstance(image, (PIL.Image.Image, list)):
            processed = self.image_processor.preprocess(image)
        else:
            if isinstance(image, torch.Tensor):
                if image.dim() == 3:
                    image = image.unsqueeze(0)
                if image.dim() != 4:
                    raise ValueError(f"Tensor must be 4D [B,C,H,W], got {image.dim()}D")
            processed = image
            
        processed = processed.to(self.device)
        return self.vae.encode(processed)

    def prepare_latent(self,
                      image: Union[PIL.Image.Image, torch.Tensor, List[PIL.Image.Image], List[torch.Tensor]],
                      num_images_per_cond: int = 1) -> torch.Tensor:
        
        latent_dist = self.encode_image(image)
        latent = latent_dist.latent_dist.sample()
        latent = self.vae.config.scaling_factor * latent

        # Duplicate each input num_images_per_cond times
        latents = latent.repeat_interleave(num_images_per_cond, dim=0)
        return latents

    def decode_latent(self, latent: torch.Tensor) -> Union[PIL.Image.Image, List[PIL.Image.Image]]:
        latents = 1 / 0.18215 * latent
        with torch.no_grad():
            images = self.vae.decode(latents).sample
            # Detach the tensor before postprocessing
            images = images.detach()
        
        processed_images = self.image_processor.postprocess(images)
        return processed_images[0] if len(processed_images) == 1 else processed_images