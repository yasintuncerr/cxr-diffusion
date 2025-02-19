import os 
import math

from typing import Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


from ..utils import checkpoint_path_corrector,create_destination_path

@dataclass
class ProjectionConfig:
    input_dim: Tuple[int, ...] = (1,32,128)
    output_dim: Tuple[int, ...] = (1,768)
    compression_factor: int = 2

class BidirectionalProjection(nn.Module):
    def __init__(self, config:ProjectionConfig):
        super().__init__()
        self.config = config

        if len(config.input_dim) != 3:
            raise ValueError("input_dim must have 3 elements, 1x32x128")
        
        if len(config.output_dim) != 2:
            raise ValueError("output_dim must have 2 elements, 1x768")
        
        self.input_lenght = math.prod(config.input_dim)
        self.output_lenght = math.prod(config.output_dim)

        self.hidden_lenght = (self.input_lenght + self.output_lenght) // config.compression_factor

        self.forward_projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.input_lenght, self.hidden_lenght),
            nn.LayerNorm(self.hidden_lenght),
            nn.GELU(),
            nn.Linear(self.hidden_lenght, self.output_lenght)
        )

        self.backward_projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.output_lenght, self.hidden_lenght),
            nn.LayerNorm(self.hidden_lenght),
            nn.GELU(),
            nn.Linear(self.hidden_lenght, self.input_lenght)
        )

    def forward(self, x:torch.Tensor):
        batch_size = x.size(0)

        # Forward pass: input -> embedding
        forward = self.forward_projection(x)
        forward = forward.view(batch_size, *self.config.output_dim)

        # Backward pass: embedding -> reconstruction
        backward = self.backward_projection(forward.view(batch_size, -1))
        backward = backward.view(batch_size, *self.config.input_dim)

        return forward, backward
    
    @classmethod
    def _load_ckpt(cls, checkpoint_path:str = None, sub_folder:str= None):
        config = None
        state_dict = None

        if checkpoint_path is not None:
            try:
                checkpoint_path = checkpoint_path_corrector(checkpoint_path, sub_folder=sub_folder)
            except ValueError as e:
                print(e)
                return None
            state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if state_dict.get("config", None) is not None:
                state_config = ProjectionConfig()
                state_config.input_dim = state_dict["config"].get("input_dim")
                state_config.output_dim = state_dict["config"].get("output_dim")
                state_config.compression_factor = state_dict["config"].get("compression_factor")

                config = state_config

        if config is None:
            raise ValueError("config not found in the checkpoint")

        return state_dict, config

    @classmethod
    def from_pretrained(cls, config:ProjectionConfig=None, checkpoint_path:str=None, sub_folder = None, device:str = "cpu"):
        state_dict, config = cls._load_ckpt(checkpoint_path, sub_folder=sub_folder)
        
        model = cls(config)

        if state_dict is not None:
            model.load_state_dict(state_dict["model_state_dict"])
            model = model.to(device)

        return model
    
    @classmethod
    def ForwardProjection(cls, config:ProjectionConfig=None, checkpoint_path:str=None, sub_folder = None, device:str = "cpu"):
        state_dict, config = cls._load_ckpt(checkpoint_path, sub_folder=sub_folder)
    
        model = cls(config)

        class ForwardProjection(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.forward_projection = model.forward_projection

            def forward(self, x):
                return self.forward_projection(x)
            
        forward_model = ForwardProjection(model)

        if state_dict is not None:
            # Create a new state dict with the correct keys
            new_state_dict = {}
            for key, value in state_dict['model_state_dict'].items():
                if key.startswith('forward_projection.'):
                    # Keep the key as is since our model now uses forward_projection directly
                    new_state_dict[key] = value
            
            # Load the state dict
            try:
                forward_model.load_state_dict(new_state_dict)
            except Exception as e:
                print(f"Warning: Failed to load state dict with error: {e}")
                print("Attempting to load with strict=False...")
                forward_model.load_state_dict(new_state_dict, strict=False)
            
            forward_model = forward_model.to(device)

        return forward_model
    
    def save_checkpoint(self, checkpoint_path:str, metadata:dict=None):
        state_dict = {
            "config": self.config,
            "model_state_dict": self.state_dict()
        }

        if metadata is not None:
            state_dict.update(metadata)

        sub_folder = "projection"
        path = create_destination_path(checkpoint_path, sub_folder=sub_folder)

        if checkpoint_path is None:
            raise ValueError("Invalid path type for {checkpoint_path}")
        
        torch.save(state_dict, path)
        return path