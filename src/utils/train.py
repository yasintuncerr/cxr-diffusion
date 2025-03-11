import os
import torch
import numpy as np
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from accelerate import Accelerator
from tqdm.auto import tqdm
import math
from PIL import Image
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR

from ..pipelines import VaeProcessor

@dataclass
class TrainConfig:
    batch_size: int = 32
    num_workers: int = 4
    mixed_precision: bool = True
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5  # Added weight decay
    num_epochs: int = 10
    save_interval: int = 1
    gradient_accumulation_steps: int = 1
    output_dir: str = "output"
    log_interval: int = 32
    num_validation_samples: int = 4
    image_size: int = 512  # Size of generated images
    log_with: str = "tensorboard"  # Options: "tensorboard", "wandb", "all"
    scheduler_type: str = "cosine"  # Options: "cosine", "onecycle", "none"
    warmup_steps: int = 500  # Warmup steps for schedulers
    early_stopping_patience: int = 5  # Early stopping patience (0 to disable)
    # Time-step weighting config
    use_timestep_weights: bool = True  # Enable timestep importance weighting
    high_noise_weight: float = 0.8  # Weight for high-noise timesteps
    low_noise_weight: float = 1.2  # Weight for low-noise timesteps


class Unet2DConditionalTrainer():
    def __init__(self,
                unet,
                train_config: TrainConfig,
                noise_scheduler,
                optimizer,
                ):
        
        self.unet = unet
        self.config = train_config
        self.noise_scheduler = noise_scheduler
        self.optimizer = optimizer
        self.scheduler = None
        self.early_stopping_counter = 0
        self.vae_processor = None

        # Initialize training state variables
        self.current_epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
        # Initialize learning rate scheduler based on config
        self._setup_scheduler()
        
        # Initialize Accelerator with logging capability
        self.accelerator = Accelerator(
            mixed_precision="fp16" if self.config.mixed_precision else None,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            project_dir=os.path.join(self.config.output_dir, "logs"),
            log_with=self.config.log_with
        )   

        if self.accelerator.is_main_process:
            os.makedirs(self.config.output_dir, exist_ok=True)
            os.makedirs(os.path.join(self.config.output_dir, "samples"), exist_ok=True)
            os.makedirs(os.path.join(self.config.output_dir, "checkpoints"), exist_ok=True)
            os.makedirs(os.path.join(self.config.output_dir, "best"), exist_ok=True)
            os.makedirs(self.accelerator.project_dir, exist_ok=True)
            self.accelerator.print(f"Log directory is {self.accelerator.project_dir}")
            self.accelerator.print(f"Output directory is {self.config.output_dir}")
            
            # Initialize trackers for logging
            self.accelerator.init_trackers(
                project_name="unet-training",
                config=vars(self.config),
                init_kwargs={"tensorboard": {"flush_secs": 30}}
            )
            
        # Create timestep weights if enabled
        self.timestep_weights = self._create_timestep_weights() if self.config.use_timestep_weights else None

    def _setup_scheduler(self):
        """Setup learning rate scheduler based on config"""
        if self.config.scheduler_type == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer, 
                T_max=self.config.num_epochs,
                eta_min=self.config.learning_rate / 10
            )
            return
        
        elif self.config.scheduler_type == "onecycle":
            # Calculate total steps for OneCycleLR
            # This will be updated after dataloader preparation
            self.scheduler = 'onecycle_pending'
            return
        
        # No scheduler
        self.scheduler = None

    def _create_timestep_weights(self):
        """Create weights for different timesteps to focus on different noise levels"""
        num_timesteps = self.noise_scheduler.config.num_train_timesteps
        # Linear interpolation between high and low noise weights
        weights = torch.linspace(
            self.config.high_noise_weight, 
            self.config.low_noise_weight, 
            num_timesteps
        )
        return weights

    def save(self, subdir: Optional[str] = None, prefix: str = "", include_optimizer: bool = True):
        """Save the UNet model and training state to disk"""
        unwrapped_model = self.accelerator.unwrap_model(self.unet)
        if self.accelerator.is_main_process:
            # Determine save directory
            if subdir is not None:
                save_dir = os.path.join(self.config.output_dir, subdir)
            else:
                save_dir = self.config.output_dir
                
            os.makedirs(save_dir, exist_ok=True)
            
            # Save model
            model_path = os.path.join(save_dir, f"{prefix}")
            os.makedirs(model_path, exist_ok=True)
            self.accelerator.print(f"Saving model to {model_path}")
            unwrapped_model.save_pretrained(model_path)
            
            # Save optimizer state
            if include_optimizer:
                optimizer_path = os.path.join(save_dir,prefix, "optimizer.pt")
                torch.save(self.optimizer.state_dict(), optimizer_path)
                
                # Save scheduler state if exists
                if self.scheduler is not None and not isinstance(self.scheduler, str):
                    scheduler_path = os.path.join(save_dir, prefix, "scheduler.pt")
                    torch.save(self.scheduler.state_dict(), scheduler_path)
                    
            # Save training state (for resuming)
            training_state = {
                'epoch': self.current_epoch,
                'global_step': self.global_step,
                'best_loss': self.best_loss,
                'early_stopping_counter': self.early_stopping_counter
            }
            state_path = os.path.join(save_dir, prefix,"training_state.pt")
            torch.save(training_state, state_path)
    
    def load(self, path: str, load_optimizer: bool = True):
        """Load model and training state from path"""
        # Check if path exists
        if not os.path.exists(path):
            self.accelerator.print(f"Cannot load model: {path} does not exist")
            return False
            
        try:
            
            # Load model weights
            self.unet.from_pretrained(path)
            self.accelerator.print(f"Loaded model from {path}")
            
            files = os.listdir(path)
            # Load optimizer if requested
            
            if "optimizer.pt" in files:
                optimizer_path = os.path.join(path, "optimizer.pt")
                self.optimizer.load_state_dict(torch.load(optimizer_path))
                self.accelerator.print(f"Loaded optimizer from {optimizer_path}")
            else:
                return False

            if "scheduler.pt" in files:
                scheduler_path = os.path.join(path, "scheduler.pt")
                self.scheduler.load_state_dict(torch.load(scheduler_path))
                self.accelerator.print(f"Loaded scheduler from {scheduler_path}")
            else:
                return False

            if "training_state.pt" in files:
                state_path = os.path.join(path, "training_state.pt")
                training_state = torch.load(state_path)
                self.current_epoch = training_state.get('epoch', 0)
                self.global_step = training_state.get('global_step', 0)
                self.best_loss = training_state.get('best_loss', float('inf'))
                self.early_stopping_counter = training_state.get('early_stopping_counter', 0)
                self.accelerator.print(f"Resuming from epoch {self.current_epoch}, global step {self.global_step}")    
            else:
                return False
                
            return True
        except Exception as e:
            self.accelerator.print(f"Error loading model: {e}")
            import traceback
            traceback.print_exc()
            return False

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
                
                # Start from random noise with a fixed seed for reproducibility
                generator = torch.Generator(device=device).manual_seed(i)
                latents = torch.randn(
                    (1, 4, self.config.image_size // 8, self.config.image_size // 8),
                    device=device,
                    generator=generator,
                )
                
                # Set timesteps
                self.noise_scheduler.set_timesteps(num_inference_steps, device=device)
                timesteps = self.noise_scheduler.timesteps
                
                # Denoising loop with progress bar
                for t in tqdm(timesteps, desc=f"Generating sample {i+1}/{batch_size}", disable=not self.accelerator.is_main_process):
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
                    # Decode using VAE processor
                    image = self.vae_processor.decode_latent(latents)[0]
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
        
        # Log the image to trackers if they're initialized
        if self.accelerator.trackers:
            for tracker in self.accelerator.trackers:
                if hasattr(tracker, 'add_image'):
                    tracker.add_image(
                        f"sample_grid_epoch_{epoch}", 
                        np.array(grid_image).transpose(2, 0, 1), 
                        epoch
                    )
        
        return grid_image
    
    def evaluate(self, validation_dataloader):
        """Evaluate model on validation dataset"""
        self.unet.eval()
        val_loss = 0.0
        val_psnr = 0.0
        
        with torch.no_grad():
            for step, (latents, cond_embed) in enumerate(validation_dataloader):
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
                noise_pred = self.unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=cond_embed,
                ).sample
                
                # Calculate loss
                loss = torch.nn.functional.mse_loss(noise_pred, noise)
                
                # Calculate PSNR
                psnr = 20 * torch.log10(torch.max(noise) / torch.sqrt(loss))
                
                val_loss += loss.item()
                val_psnr += psnr.item()
        
        # Calculate averages
        avg_val_loss = val_loss / len(validation_dataloader)
        avg_val_psnr = val_psnr / len(validation_dataloader)
        
        return avg_val_loss, avg_val_psnr

    def train(self, dataloader, validation_dataloader=None, validation_samples=None, resume_from=None):
        """Train the UNet model with sample generation at save intervals"""
        # If resume_from is specified, load checkpoint
        if resume_from:
            if self.load(resume_from):
                self.accelerator.print(f"Successfully loaded checkpoint from {resume_from}")
                self.accelerator.print(f"Resuming from epoch {self.current_epoch+1}, step {self.global_step}")
            else:
                self.accelerator.print(f"Failed to load checkpoint from {resume_from}, starting fresh")
                self.current_epoch = 0
                self.global_step = 0
                self.best_loss = float('inf')
                self.early_stopping_counter = 0

        # Setup OneCycleLR if selected (needs total_steps)
        if self.scheduler == 'onecycle_pending':
            total_steps = len(dataloader) * (self.config.num_epochs - self.current_epoch) // self.config.gradient_accumulation_steps
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=self.config.learning_rate,
                total_steps=total_steps,
                pct_start=0.1,  # Warmup percentage
                div_factor=25,  # Initial lr = max_lr/div_factor
                final_div_factor=1000,  # Final lr = initial_lr/final_div_factor
            )

        # Prepare model and dataloader with accelerator
        self.unet, self.optimizer, dataloader = self.accelerator.prepare(
            self.unet, self.optimizer, dataloader
        )

        # Prepare validation dataloader if provided
        if validation_dataloader is not None:
            validation_dataloader = self.accelerator.prepare(validation_dataloader)
        if self.timestep_weights is not None:
            self.timestep_weights = self.timestep_weights.to(self.accelerator.device)

        # Progress bar for epochs
        remaining_epochs = self.config.num_epochs - self.current_epoch
        progress_bar = tqdm(range(remaining_epochs), disable=not self.accelerator.is_local_main_process)
        progress_bar.set_description(f"Epochs")

        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.current_epoch = epoch
            self.unet.train()

            epoch_loss = 0.0
            epoch_psnr = 0.0

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
                    if self.timestep_weights is not None:
                        # Apply timestep-specific weights to loss
                        weights = self.timestep_weights[timesteps]
                        element_wise_loss = torch.nn.functional.mse_loss(
                            noise_pred, noise, reduction="none"
                        ).mean(dim=[1, 2, 3])
                        loss = (element_wise_loss * weights).mean()
                    else:
                        loss = torch.nn.functional.mse_loss(noise_pred, noise)

                    # Backpropagate
                    self.accelerator.backward(loss)

                    if step % self.config.gradient_accumulation_steps == self.config.gradient_accumulation_steps - 1 or step == len(dataloader) - 1:
                        if hasattr(self.accelerator, "clip_grad_norm_"):
                            self.accelerator.clip_grad_norm_(self.unet.parameters(), 1.0)

                        # Update weights
                        self.optimizer.step()

                        # Update learning rate for OneCycleLR which updates per step
                        if isinstance(self.scheduler, OneCycleLR):
                            self.scheduler.step()

                        self.optimizer.zero_grad()

                # Update progress and logging
                if self.accelerator.is_main_process:
                    step_progress_bar.update(1)
                    epoch_loss += loss.item()
                    self.global_step += 1

                    # Calculate PSNR (Peak Signal-to-Noise Ratio)
                    with torch.no_grad():
                        psnr = 20 * torch.log10(torch.max(noise) / torch.sqrt(loss))
                        epoch_psnr += psnr.item()

                    # Log detailed metrics at regular intervals
                    if step % self.config.log_interval == 0 or step == len(dataloader) - 1:
                        # Get current learning rate
                        current_lr = self.optimizer.param_groups[0]['lr']

                        # Update progress bar with metrics
                        step_progress_bar.set_postfix(
                            loss=f"{loss.item():.4f}", 
                            psnr=f"{psnr.item():.2f}",
                            lr=f"{current_lr:.6f}"
                        )

                        # Log metrics to all trackers
                        self.accelerator.log({
                            "train/loss": loss.item(),
                            "train/psnr": psnr.item(),
                            "train/lr": current_lr,
                            "train/epoch": epoch,
                            "train/global_step": self.global_step,
                        }, step=self.global_step)

            step_progress_bar.close()  # Close the progress bar

            # Update learning rate scheduler (except for OneCycleLR which updates per step)
            if self.scheduler is not None and not isinstance(self.scheduler, OneCycleLR):
                self.scheduler.step()

            # Calculate average epoch metrics
            avg_loss = epoch_loss / len(dataloader)
            avg_psnr = epoch_psnr / len(dataloader)

            # Run validation if validation dataloader is provided
            val_metrics = {}
            if validation_dataloader is not None:
                self.accelerator.print(f"Running validation for epoch {epoch+1}...")
                val_loss, val_psnr = self.evaluate(validation_dataloader)
                val_metrics = {
                    "val/loss": val_loss,
                    "val/psnr": val_psnr,
                    "val/epoch": epoch,
                }
                self.accelerator.print(f"Validation - Loss: {val_loss:.6f}, PSNR: {val_psnr:.2f}")

                # Use validation loss for model selection
                comparison_loss = val_loss
            else:
                # Use training loss for model selection if no validation
                comparison_loss = avg_loss

            if self.accelerator.is_main_process:
                progress_bar.update(1)
                progress_bar.set_postfix(
                    avg_loss=avg_loss, 
                    avg_psnr=avg_psnr,
                    **({f"val_loss": val_metrics.get("val/loss", 0)} if val_metrics else {})
                )

                # Log epoch metrics
                epoch_log = {
                    "train/epoch_loss": avg_loss,
                    "train/epoch_psnr": avg_psnr,
                    "train/epoch": epoch,
                }
                if val_metrics:
                    epoch_log.update(val_metrics)

                self.accelerator.log(epoch_log, step=self.global_step)

                # Print epoch summary
                self.accelerator.print(f"Epoch {epoch+1}/{self.config.num_epochs} - "
                                     f"Avg loss: {avg_loss:.6f}, "
                                     f"Avg PSNR: {avg_psnr:.2f}")

                # Save if this is the best model
                if comparison_loss < self.best_loss:
                    self.best_loss = comparison_loss
                    self.save(subdir="best", prefix="best", include_optimizer=True)
                    self.accelerator.print(f"New best model with loss: {self.best_loss:.6f}")
                    # Reset early stopping counter
                    self.early_stopping_counter = 0
                else:
                    # Increment early stopping counter
                    self.early_stopping_counter += 1
                    self.accelerator.print(f"No improvement. Early stopping counter: {self.early_stopping_counter}/{self.config.early_stopping_patience}")

                    # Check if we should stop early
                    if (self.config.early_stopping_patience > 0 and 
                        self.early_stopping_counter >= self.config.early_stopping_patience):
                        self.accelerator.print(f"Early stopping triggered after {epoch+1} epochs")
                        break
                    
                # Save based on interval
                if (epoch + 1) % self.config.save_interval == 0:
                    self.save(subdir="checkpoints", prefix=f"epoch_{epoch+1}", include_optimizer=True)

                    # Generate samples at save interval if validation samples are provided
                    if validation_samples is not None:
                        self.accelerator.print(f"Generating validation samples for epoch {epoch+1}...")
                        sample_images = self.generate_samples(validation_samples)
                        if sample_images:
                            self.save_image_grid(sample_images, epoch+1)

        # Save final model
        self.save(prefix="final", include_optimizer=True)

        # Generate final samples
        if validation_samples is not None:
            self.accelerator.print("Generating samples with final model...")
            sample_images = self.generate_samples(validation_samples)
            if sample_images:
                self.save_image_grid(sample_images, self.config.num_epochs, filename="final_samples.png")

        # Load best model for return
        self.accelerator.print(f"Loading best model")
        self.load(os.path.join(self.config.output_dir, "best"), load_optimizer=False)

        # End logging
        self.accelerator.end_training()

        return self.accelerator.unwrap_model(self.unet)


def get_validation_samples(dataset, num_samples=4, max_seq_len=77, embed_dim=768):
    """Extract validation samples from a dataset for generating samples during training."""
    validation_cond_embeds = []
    
    # Create an empty tensor with the right dimensions
    zero_cond_embed = torch.zeros(1, max_seq_len, embed_dim)
    
    # Randomly select samples from the dataset
    dataset_size = len(dataset)
    random_indices = np.random.choice(dataset_size, min(num_samples, dataset_size), replace=False)
    
    for i in random_indices:
        # Get the sample from the dataset
        try:
            sample = dataset[i]
            if isinstance(sample, tuple) and len(sample) >= 2:
                _, cond_embed = sample[0], sample[1]
                
                # Handle different possible dimensions
                if len(cond_embed.shape) == 2:  # If shape is [seq_len, embed_dim]
                    padded_embed = zero_cond_embed.clone()
                    seq_len = min(cond_embed.shape[0], max_seq_len)
                    padded_embed[0, :seq_len, :] = cond_embed[:seq_len, :]
                    validation_cond_embeds.append(padded_embed)
                    
                elif len(cond_embed.shape) == 3:  # If shape is [batch, seq_len, embed_dim]
                    padded_embed = zero_cond_embed.clone()
                    seq_len = min(cond_embed.shape[1], max_seq_len)
                    padded_embed[0, :seq_len, :] = cond_embed[0, :seq_len, :]
                    validation_cond_embeds.append(padded_embed)
                    
                else:
                    # Use a zero embedding as fallback
                    validation_cond_embeds.append(zero_cond_embed.clone())
                    
        except Exception as e:
            # Use a zero embedding as fallback
            validation_cond_embeds.append(zero_cond_embed.clone())
    
    # Convert the list of tensors to a single tensor
    if validation_cond_embeds:
        validation_tensor = torch.cat(validation_cond_embeds, dim=0)
        return validation_tensor
    else:
        # If no valid samples were found, return a tensor of zeros
        return torch.zeros(num_samples, 1, max_seq_len, embed_dim)


def create_validation_dataloader(train_dataset, validation_split=0.1, batch_size=8, num_workers=4):
    """Create a validation dataloader from a training dataset"""
    from torch.utils.data import random_split, DataLoader
    
    # Calculate split sizes
    dataset_size = len(train_dataset)
    val_size = int(validation_split * dataset_size)
    train_size = dataset_size - val_size
    
    # Split the dataset
    train_subset, val_subset = random_split(
        train_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # Fixed seed for reproducibility
    )
    
    # Create validation dataloader
    val_dataloader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return val_dataloader, train_subset


def get_linear_warmup_cosine_decay_scheduler(optimizer, warmup_steps, total_steps, min_lr_ratio=0.1):
    """Create a custom learning rate scheduler with warmup and cosine decay"""
    from torch.optim.lr_scheduler import LambdaLR
    
    def lr_lambda(current_step):
        # Warmup phase
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        
        # Cosine decay after warmup
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return LambdaLR(optimizer, lr_lambda)