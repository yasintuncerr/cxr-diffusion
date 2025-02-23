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
class ElixrConditionedUNetConfig(PretrainedConfig):
    """Configuration class for FeatureConditionedUNet."""
    
    feature_dim: int = 4096
    unet_config: Optional[dict] = None
    projection_config: Optional[dict] = None
    model_type: str = "feature_conditioned_unet"
    
    def __init__(self, **kwargs):
        self.feature_dim = kwargs.pop("feature_dim", 4096)
        self.unet_config = kwargs.pop("unet_config", None)
        self.projection_config = kwargs.pop("projection_config", None)
        self.model_type = kwargs.pop("model_type", "feature_conditioned_unet")
        
        # Initialize parent class with remaining kwargs
        super().__init__(**kwargs)

class ElixrConditionedUNet(PreTrainedModel):
    config_class = ElixrConditionedUNetConfig
    base_model_prefix = "elixr_conditioned_unet"
    
    def __init__(self, config: ElixrConditionedUNetConfig):
        super().__init__(config)
        
        # Initialize UNet
        if config.unet_config is not None:
            self.unet = UNet2DConditionModel(**config.unet_config)
        else:
            raise ValueError("UNet config must be provided.")
        
        # Initialize projection
        projection_kwargs = {
            "input_dim": config.feature_dim,
            **(config.projection_config or {})
        }
        # Only set cross_attention_dim if not already in projection_config
        if "cross_attention_dim" not in projection_kwargs:
            projection_kwargs["cross_attention_dim"] = self.unet.config.cross_attention_dim
            
        projection_config = ElixrBProjectionConfig(**projection_kwargs)
        self.projection = ElixrBProjection(projection_config)
    
    def forward(self, x, timesteps, features, return_dict=False):
        projected_features = self.projection(features)
        return self.unet(
            x, 
            timesteps, 
            encoder_hidden_states=projected_features,
            return_dict=return_dict
        )
    
    def save_pretrained(self, save_directory, **kwargs):
        """Save both UNet and projection models."""
        # Save the main config
        self.config.unet_config = dict(self.unet.config)
        self.config.projection_config = self.projection.config.to_dict()        
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