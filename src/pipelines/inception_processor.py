import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from typing import List, Tuple, Union
from PIL import Image
import numpy as np

class InceptionProcessor(nn.Module):
    def __init__(self,
                 device = 'cuda'):
        super(InceptionProcessor, self).__init__()
        self.inception = models.inception_v3(weights ="DEFAULT")
        self.inception = self.inception.to(device)



    def __call__(self, input: torch.Tensor) -> np.ndarray:
        batch_size = input.size(0)
        with torch.no_grad():
            x = self.inception(input)
            if isinstance(x, tuple):
                x = x[0]

            x = x.cpu().numpy()
            x = x.reshape((batch_size, -1))
        
        return x
