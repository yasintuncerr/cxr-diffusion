from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig
from diffusers import UNet2DConditionModel


@dataclass
class ElixrBProjectionConfig(PretrainedConfig):
    """Configuration class for ElixrBProjection."""
    
    input_dim: int = 4096
    cross_attention_dim: int = 768
    hidden_dim: int = 1024
    num_hidden_layers: int = 2
    dropout: float = 0.1
    model_type: str = "elixr_b_projection"
    
    def __post_init__(self):
        super().__init__()


class ElixrBProjection(PreTrainedModel):
    config_class = ElixrBProjectionConfig
    base_model_prefix = "elixr_b_projection"
    
    def __init__(self, config: ElixrBProjectionConfig):
        super().__init__(config)
        
        layers = []
        current_dim = config.input_dim
        
        # Hidden layers
        for _ in range(config.num_hidden_layers):
            layers.extend([
                nn.Linear(current_dim, config.hidden_dim),
                nn.LayerNorm(config.hidden_dim),
                nn.GELU(),
                nn.Dropout(config.dropout)
            ])
            current_dim = config.hidden_dim
        
        # Output layer
        layers.extend([
            nn.Linear(current_dim, config.cross_attention_dim),
            nn.LayerNorm(config.cross_attention_dim)
        ])
        
        self.projection = nn.Sequential(*layers)
    
    def forward(self, x):
        orig_shape = x.shape
        if len(orig_shape) == 3:
            x = x.view(-1, orig_shape[-1])
        
        x = self.projection(x)
        
        if len(orig_shape) == 3:
            x = x.view(orig_shape[0], orig_shape[1], -1)
        
        return x


@dataclass
class FeatureConditionedUNetConfig(PretrainedConfig):
    """Configuration class for FeatureConditionedUNet."""
    
    feature_dim: int = 4096
    unet_config: Optional[dict] = None
    projection_config: Optional[dict] = None
    model_type: str = "feature_conditioned_unet"
    
    def __post_init__(self):
        super().__init__()


class FeatureConditionedUNet(PreTrainedModel):
    config_class = FeatureConditionedUNetConfig
    base_model_prefix = "feature_conditioned_unet"
    
    def __init__(self, config: FeatureConditionedUNetConfig):
        super().__init__(config)
        
        # Initialize UNet
        if config.unet_config is not None:
            self.unet = UNet2DConditionModel(**config.unet_config)
        else:
            raise ValueError("UNet config must be provided.")
        
        # Initialize projection
        projection_config = ElixrBProjectionConfig(
            input_dim=config.feature_dim,
            cross_attention_dim=self.unet.config.cross_attention_dim,
            **(config.projection_config or {})
        )
        self.projection = ElixrBProjection(projection_config)
    
    def forward(self, x, timesteps, features, return_dict=False):
        batch_size = x.shape[0]
        
        # Expand features if needed
        if len(features.shape) == 2:
            features = features.unsqueeze(0).repeat(batch_size, 1, 1)
        
        # Project features
        features = self.projection(features)
        
        # Pad or truncate to 77 tokens
        if features.shape[1] < 77:
            padding = torch.zeros(
                batch_size, 
                77-features.shape[1], 
                features.shape[-1],
                device=features.device
            )
            features = torch.cat([features, padding], dim=1)
        elif features.shape[1] > 77:
            print(f"Warning: features.shape[1] > 77, truncating to 77")
            features = features[:, :77, :]
        
        return self.unet(x, timesteps, encoder_hidden_states=features, return_dict=return_dict)
    
    def save_pretrained(self, save_directory, **kwargs):
        """Save both UNet and projection models."""
        # Save the main config
        self.config.unet_config = self.unet.config.to_dict()
        self.config.projection_config = self.projection.config.to_dict()
        super().save_pretrained(save_directory, **kwargs)
        
        # Save UNet and projection in subdirectories
        self.unet.save_pretrained(f"{save_directory}/unet")
        self.projection.save_pretrained(f"{save_directory}/projection")
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """Load both UNet and projection models."""
        # Load the main model and config
        model = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        
        # Load UNet and projection from subdirectories
        model.unet = UNet2DConditionModel.from_pretrained(
            f"{pretrained_model_name_or_path}/unet"
        )
        model.projection = ElixrBProjection.from_pretrained(
            f"{pretrained_model_name_or_path}/projection"
        )
        
        return model

# Example usage:
"""
# Create and save model
config = FeatureConditionedUNetConfig(
    feature_dim=4096,
    projection_config={"hidden_dim": 1024, "num_hidden_layers": 2}
)
model = FeatureConditionedUNet(config)
model.save_pretrained("path/to/save")

# Load model
loaded_model = FeatureConditionedUNet.from_pretrained("path/to/save")
"""