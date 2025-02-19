import PIL.Image
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from typing import Optional, Union

from diffusers.image_processor import VaeImageProcessor
from diffusers import AutoencoderKL, UNet2DConditionModel,DDPMScheduler
from diffusers.utils.torch_utils import randn_tensor


class SDElixrConditionedImg2ImgPipeline():
    def __init__(
        self,
        vae: AutoencoderKL,
        projection_encoder: nn.Module,
        unet: UNet2DConditionModel,
        scheduler: DDPMScheduler,
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__()
        
        self.device = device
        self.vae = vae.to(device)
        self.projection_encoder = projection_encoder.to(device)
        self.unet = unet.to(device)
        self.scheduler = scheduler
        # Create an instance of VaeImageProcessor
        self.image_processor = VaeImageProcessor()

        
        # Define default image dimensions
        self.height = 512  # Default height
        self.width = 512   # Default width
        
    @torch.no_grad()
    @torch.no_grad()
    def prepare_conditional_input(self, elixr_embed: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            embed = self.projection_encoder(elixr_embed)
            # Add a check for dimensionality
            if embed.dim() == 2:
                # Add batch dimension if missing
                embed = embed.unsqueeze(0)
            # Log shape for debugging
            print(f"Conditional input shape: {embed.shape}")
        return embed
    @torch.no_grad()
    def prepare_latent_input(self, image: PIL.Image.Image, generator=None) -> torch.Tensor:
        if image is not None:
            # Create an instance of VaeImageProcessor
            image_processor = VaeImageProcessor()
            
            # Convert image to RGB if it's not already
            if image.mode != 'RGB':
                image = image.convert('RGB')
                
            # Preprocess the image
            image = image_processor.preprocess(image)
            image = image.to(self.device)
            
            with torch.no_grad():
                # The encode method returns an object with different naming
                encoder_output = self.vae.encode(image)
                latents = encoder_output.latent_dist.sample()
                
            latents = self.vae.config.scaling_factor * latents
            latents = torch.cat([latents], dim=0)
    
            shape = latents.shape
            # Convert device to torch.device if it's a string
            device = torch.device(self.device) if isinstance(self.device, str) else self.device
            noise = randn_tensor(shape, generator=generator, device=device)
    
            # Use the first timestep for initial noise addition
            latents = self.scheduler.add_noise(latents, noise, self.scheduler.timesteps[0])
    
        else:
            # Convert device to torch.device if it's a string
            device = torch.device(self.device) if isinstance(self.device, str) else self.device
            latents = torch.randn(
                (1, self.unet.config.in_channels, self.height // 8, self.width // 8),
                generator=generator,
                device=device
            )
            latents = latents * self.scheduler.init_noise_sigma
    
        return latents
    @torch.no_grad()
    def decode_latent_input(self, latent: torch.Tensor) -> PIL.Image.Image:
        latents = 1 / 0.18215 * latent

        with torch.no_grad():
            img = self.vae.decode(latents).sample

        img = (img / 2 + 0.5).clamp(0, 1).squeeze(0)
        img = (img.permute(1, 2, 0) * 255).cpu().to(torch.uint8).numpy()
        image = PIL.Image.fromarray(img)

        return image
    
    @torch.no_grad()
    def __call__(
        self,
        image: PIL.Image.Image = None,
        elixr_embed: torch.Tensor = None,
        num_inference_steps: int = 25,
        guidance_scale: float = 1.0,
        generator: Optional[torch.Generator] = None
        ):

        # Only do classifier guidance if we have an embedding and guidance_scale > 1.0
        do_classifier_guidance = guidance_scale > 1.0 and elixr_embed is not None

        # Set scheduler timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)

        # Prepare latents
        latents = self.prepare_latent_input(image, generator)

        # Handle conditional input
        if elixr_embed is not None:
            cond_input = self.prepare_conditional_input(elixr_embed.to(self.device))
            if do_classifier_guidance:
                uncond_input = torch.zeros_like(cond_input).to(self.device)
                cond_input = torch.cat([uncond_input, cond_input])
        else:
            # If no condition is provided, create zero tensor with correct shape
            batch_size = latents.shape[0]
            cross_attention_dim = self.unet.config.cross_attention_dim
            cond_input = torch.zeros(batch_size, 77, cross_attention_dim, device=self.device)

        # Denoising loop
        for t in tqdm(self.scheduler.timesteps, desc="Processing"):
            latent_model_input = self.scheduler.scale_model_input(latents, t)

            if do_classifier_guidance:
                latent_model_input = torch.cat([latent_model_input] * 2)

            # Predict noise
            with torch.no_grad():
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=cond_input,
                    return_dict=False
                )[0]

            # Perform guidance if needed (only when we have an embedding)
            if do_classifier_guidance:
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            # Scheduler step
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        return self.decode_latent_input(latents)