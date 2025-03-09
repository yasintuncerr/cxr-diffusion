import os
import pickle
from ..pipelines import CLIPTextProcessor, CLIPImageProcessor, VaeProcessor, InceptionProcessor
from torch.utils.data import DataLoader
from typing import Union, List
from abc import ABC, abstractmethod

class FeatureExtractor:
    def __init__(self,
                 processor: Union[CLIPTextProcessor, CLIPImageProcessor, VaeProcessor, InceptionProcessor],
                 dataloader: DataLoader,
                 save_dir: str = None,
                 save_name: str = None,
                 chunk_size: int = 10000):
        self.processor = processor
        self.dataloader = dataloader
        self.save_dir = save_dir
        self.save_name = save_name
        self.chunk_size = chunk_size

        if not self.save_dir or not self.save_name:
            raise ValueError("save_dir and save_name must be provided")
        
        os.makedirs(self.save_dir, exist_ok=True)
    
    @abstractmethod
    def extract(self):
        pass


    def save(self, data:dict):
        if self.save_dir is None or self.save_name is None:
            return
        with open(f"{self.save_dir}/{self.save_name}.pkl", "wb") as f:
            pickle.dump(data, f)

    def run(self):
        data = {}
        chunk_idx = -1
        for i, batch in enumerate(self.dataloader):
            output = self.extract(batch)
            data.update(output)
            if i % self.chunk_size == 0:
                chunk_idx = i // self.chunk_size
                self.save_name = f"{self.save_name}_{chunk_idx}"
                self.save(data)
                data = {}
                
        if chunk_idx != -1:
            self.save_name = f"{self.save_name}_{chunk_idx}"
        
        if data:
            self.save(data)


class ClipTextFeatureExtractor(FeatureExtractor):
    def __init__(self,
                 processor: CLIPTextProcessor,
                 dataloader: DataLoader,
                 save_dir: str = None,
                 save_name: str = None,
                 chunk_size: int = 10000):
        super().__init__(processor, dataloader, save_dir, save_name, chunk_size)
    
    
    def extract(self, batch):
        image_idx, texts = batch
        data = {}
        for i, idx in enumerate(image_idx):
            data[idx] = self.processor.encode_text(texts[i])
        return data
    

class ClipVisionFeatureExtractor(FeatureExtractor):
    def __init__(self,
                 processor: CLIPImageProcessor,
                 dataloader: DataLoader,
                 save_dir: str = None,
                 save_name: str = None,
                 chunk_size: int = 10000):
        super().__init__(processor, dataloader, save_dir, save_name, chunk_size)
    
    
    def extract(self, batch):
        image_idx, images = batch
        image_embeds, _ = self.processor.encode_image(images)
        data = {}
        for i, idx in enumerate(image_idx):
            data[idx] = image_embeds[i]
        return data


class VaeFeatureExtractor(FeatureExtractor):
    def __init__(self,
                 processor: VaeProcessor,
                 dataloader: DataLoader,
                 save_dir: str = None,
                 save_name: str = None,
                 chunk_size: int = 10000):
        super().__init__(processor, dataloader, save_dir, save_name, chunk_size)
    

    def extract(self, batch):
        image_idx, images = batch
        output = self.processor.encode_image(images)
        data = {}
        for i, idx in enumerate(image_idx):
            data[idx] = output[i]
        return data