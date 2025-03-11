import os
import pickle
from ..pipelines import CLIPTextProcessor, ClipVisionProcessor, VaeProcessor, InceptionProcessor
from torch.utils.data import DataLoader
from typing import Union, List
from abc import ABC, abstractmethod
from tqdm.auto import tqdm

class FeatureExtractor:
    def __init__(self,
                 processor: Union[CLIPTextProcessor, ClipVisionProcessor, VaeProcessor, InceptionProcessor],
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


    def save(self, data: dict):
        if self.save_dir is None or self.save_name is None:
            return
        print(f"Saving data to {self.save_dir}/{self.save_name}.pkl")
        with open(f"{self.save_dir}/{self.save_name}.pkl", "wb") as f:
            pickle.dump(data, f)


    def run(self):
        data = {}
        chunk_idx = -1

        print(f"Dataloader length: {len(self.dataloader)}")  # Dataloader uzunluğunu kontrol et
        save_name = self.save_name

        with tqdm(total=len(self.dataloader), desc="Processing Batches") as pbar:
            for i, batch in enumerate(self.dataloader):
                output = self.extract(batch)
                data.update(output)

                if len(data.keys()) >= self.chunk_size:
                    chunk_idx +=1
                    self.save_name = f"{save_name}_{chunk_idx:03}"
                    print(f"Saving chunk {chunk_idx} on {self.save_dir}/{self.save_name}.pkl")
                    self.save(data)
                    data = {}
                
                pbar.update(1)

            if chunk_idx != -1:
                chunk_idx += 1
                self.save_name = f"{save_name}_{chunk_idx:03}"

            print(f"Final data: {data}")
            if data:
                print("Saving final data.")
                self.save(data)
            else:
                print("Final data is empty.")


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
                 processor: ClipVisionProcessor,
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
        output = self.processor.encode(images)
        data = {}
        for i, idx in enumerate(image_idx):
            data[idx] = output[i]
        return data