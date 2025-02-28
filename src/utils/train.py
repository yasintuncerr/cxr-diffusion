import os
import torch
import numpy as np
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from accelerate import Accelerator
from tqdm.notebook import tqdm
import copy
from PIL import Image
import math

from ..pipelines import VaeProcessor

@dataclass
class TrainConfig:
    batch_size: int = 32
    num_workers: int = 4
    mixed_precision: bool = True
    learning_rate: float = 1e-4
    num_epochs: int = 10
    save_interval: int = 1
    num_train_steps: Optional[int] = None
    gradient_accumulation_steps: int = 1
    output_dir: str = "output"
    log_interval: int = 32
    num_validation_samples: int = 4
    image_size: int = 512  # Size of generated images


class Unet2DConditionalTrainer():
    def __init__(self,
                unet,
                train_config: TrainConfig,
                noise_scheduler,
                optimizer,
                vae_processor=None
                ):
        
        self.unet = unet
        self.config = train_config  # Renamed train_config to config for consistency
        self.noise_scheduler = noise_scheduler
        self.optimizer = optimizer
        self.vae_processor = vae_processor
        self.accelerator = Accelerator(
            mixed_precision="fp16" if self.config.mixed_precision else None,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            project_dir=os.path.join(self.config.output_dir, "logs")
        )   

        if self.accelerator.is_main_process:
            os.makedirs(self.config.output_dir, exist_ok=True)
            os.makedirs(os.path.join(self.config.output_dir, "samples"), exist_ok=True)
            os.makedirs(os.path.join(self.config.output_dir, "checkpoints"), exist_ok=True)
            os.makedirs(os.path.join(self.config.output_dir, "best"), exist_ok=True)
            os.makedirs(self.accelerator.project_dir, exist_ok=True)
            self.accelerator.print(f"Log directory is {self.accelerator.project_dir}")
            self.accelerator.print(f"Output directory is {self.config.output_dir}")

    def save(self, subdir: Optional[str] = None, prefix: str = ""):
        """Save the UNet model to disk"""
        unwrapped_model = self.accelerator.unwrap_model(self.unet)
        if self.accelerator.is_main_process:
            if subdir is not None:
                save_dir = os.path.join(self.config.output_dir, subdir)
                os.makedirs(save_dir, exist_ok=True)
                path = os.path.join(save_dir, f"{prefix}.safetensors")
            else:
                path = os.path.join(self.config.output_dir, f"{prefix}.safetensors")
                
            self.accelerator.print(f"Saving model to {path}")
            unwrapped_model.save_pretrained(path)
            
            # Save optimizer state
            optimizer_path = os.path.join(self.config.output_dir, f"{prefix}_optimizer.pt")
            torch.save(self.optimizer.state_dict(), optimizer_path)

    def generate_samples(self, validation_cond_embeds, num_inference_steps=50):
        """Generate samples from conditioning embeddings"""
        # If no VAE processor is available, try to create one
        if self.vae_processor is None:
            try:
                self.vae_processor = VaeProcessor(device=self.accelerator.device)
                self.accelerator.print("Initialized VaeProcessor for sample generation")
            except Exception as e:
                self.accelerator.print(f"Warning: Could not initialize VaeProcessor: {e}")
                return None
            
        self.unet.eval()
        device = self.accelerator.device
        
        # Prepare for batch processing
        batch_size = validation_cond_embeds.shape[0]
        sample_images = []
        
        with torch.no_grad():
            for i in range(batch_size):
                # Get single conditioning embedding
                cond_embed = validation_cond_embeds[i:i+1].to(device)
                
                # Start from random noise
                latents = torch.randn(
                    (1, 4, self.config.image_size // 8, self.config.image_size // 8),
                    device=device,
                )
                
                # Set timesteps
                self.noise_scheduler.set_timesteps(num_inference_steps, device=device)
                timesteps = self.noise_scheduler.timesteps
                
                # Denoising loop
                for t in timesteps:
                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = latents
                    latent_model_input = self.noise_scheduler.scale_model_input(latent_model_input, t)
                    
                    # predict the noise residual
                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=cond_embed
                    ).sample
                    
                    # compute the previous noisy sample x_t -> x_t-1
                    latents = self.noise_scheduler.step(noise_pred, t, latents).prev_sample
                
                # Decode the image
                try:
                    # Standard approach using the 0.18215 scale factor for Stable Diffusion
                    latents = 1 / 0.18215 * latents
                    image = self.vae_processor.decode_latent(latents)
                    sample_images.append(image)
                except Exception as e:
                    self.accelerator.print(f"Error decoding latent: {e}")
                
        return sample_images
    
    def save_image_grid(self, images, epoch, filename=None):
        """Save a grid of images"""
        if not images:
            return None
            
        # Determine grid size
        num_images = len(images)
        grid_size = math.ceil(math.sqrt(num_images))
        
        # Create a blank grid
        grid_width = grid_size * images[0].width
        grid_height = grid_size * images[0].height
        grid_image = Image.new('RGB', (grid_width, grid_height), color='white')
        
        # Paste images into grid
        for i, img in enumerate(images):
            row = i // grid_size
            col = i % grid_size
            grid_image.paste(img, (col * img.width, row * img.height))
        
        # Save the grid
        if filename is None:
            filename = f"samples_epoch_{epoch}.png"
            
        save_path = os.path.join(self.config.output_dir, "samples", filename)
        grid_image.save(save_path)
        self.accelerator.print(f"Saved sample grid to {save_path}")
        
        return grid_image

    def train(self, dataloader, validation_samples=None):
        """Train the UNet model with sample generation at save intervals
        
        Args:
            dataloader: Training data loader with (latents, cond_embed) pairs
            validation_samples: List of conditioning embeddings for sample generation
        """
        self.unet, self.optimizer, dataloader = self.accelerator.prepare(
            self.unet, self.optimizer, dataloader
        )

        global_step = 0
        best_loss = float('inf')

        # Progress bar for epochs
        progress_bar = tqdm(range(self.config.num_epochs), disable=not self.accelerator.is_local_main_process)
        progress_bar.set_description(f"Epochs")

        for epoch in range(self.config.num_epochs):
            self.unet.train()
            epoch_loss = 0.0

            # Progress bar for steps
            step_progress_bar = tqdm(total=len(dataloader), disable=not self.accelerator.is_local_main_process)
            step_progress_bar.set_description(f"Epoch {epoch+1}/{self.config.num_epochs}")

            for step, (latents, cond_embed) in enumerate(dataloader):
                latents = latents.to(self.accelerator.device)
                cond_embed = cond_embed.to(self.accelerator.device)

                # Reshape conditioning embeddings if needed
                if cond_embed.dim() == 4:
                    cond_embed = cond_embed.squeeze(1)

                # Generate random noise
                noise = torch.randn_like(latents)

                # Sample random timesteps
                timesteps = torch.randint(
                    0,
                    self.noise_scheduler.config.num_train_timesteps,
                    (latents.shape[0],),
                    device=latents.device
                ).long()

                # Add noise to latents according to noise schedule
                noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

                # Predict the noise
                with self.accelerator.accumulate(self.unet):
                    noise_pred = self.unet(
                        noisy_latents,
                        timesteps, 
                        encoder_hidden_states=cond_embed,
                    ).sample

                    # Calculate loss
                    loss = torch.nn.functional.mse_loss(noise_pred, noise)

                    # Backpropagate
                    self.accelerator.backward(loss)

                    # Update weights
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                # Update progress and logging
                if self.accelerator.is_main_process:
                    step_progress_bar.update(1)
                    epoch_loss += loss.item()
                    global_step += 1

                    # Log detailed metrics at regular intervals
                    if step % self.config.log_interval == 0 or step == len(dataloader) - 1:
                        # Calculate PSNR (Peak Signal-to-Noise Ratio)
                        with torch.no_grad():
                            psnr = 20 * torch.log10(torch.max(noise) / torch.sqrt(loss))

                        # Update progress bar with metrics
                        step_progress_bar.set_postfix(
                            loss=f"{loss.item():.4f}", 
                            psnr=f"{psnr.item():.2f}",
                            lr=f"{self.optimizer.param_groups[0]['lr']:.6f}"
                        )

                        # Print detailed log
                        self.accelerator.print(
                            f"Epoch {epoch+1}/{self.config.num_epochs}, "
                            f"Step {step}/{len(dataloader)}, "
                            f"Loss: {loss.item():.6f}, "
                            f"PSNR: {psnr.item():.2f}, "
                            f"LR: {self.optimizer.param_groups[0]['lr']:.6f}"
                        )
                        

            # Calculate average epoch loss
            avg_loss = epoch_loss / len(dataloader)
            if self.accelerator.is_main_process:
                progress_bar.update(1)
                progress_bar.set_postfix(avg_loss=avg_loss)

                # Print epoch summary
                self.accelerator.print(f"Epoch {epoch+1}/{self.config.num_epochs} - Average loss: {avg_loss:.6f}")

                # Save if this is the best model
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    self.save(subdir="best", prefix="best_model")
                    self.accelerator.print(f"New best model with loss: {best_loss:.6f}")
                    

                # Save based on interval
                if (epoch + 1) % self.config.save_interval == 0:
                    self.save(subdir="checkpoints", prefix=f"epoch_{epoch+1}")
                    
                    # Generate samples at save interval if validation samples are provided
                    if validation_samples is not None:
                        self.accelerator.print(f"Generating validation samples for epoch {epoch+1}...")
                        sample_images = self.generate_samples(validation_samples)
                        if sample_images:
                            self.save_image_grid(sample_images, epoch+1)

        # Save final model
        self.save(prefix="final_model")
        
        # Generate final samples
        if validation_samples is not None:
            self.accelerator.print("Generating samples with final model...")
            sample_images = self.generate_samples(validation_samples)
            if sample_images:
                self.save_image_grid(sample_images, self.config.num_epochs, filename="final_samples.png")

        return self.accelerator.unwrap_model(self.unet)