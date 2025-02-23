import torch
from transformers import CLIPTokenizer, CLIPTextModel, CLIPVisionModel, CLIPProcessor
from typing import Union, List
from PIL import Image

class CLIPTextProcessor:
    def __init__(self,
            tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14"),
            text_encoder: CLIPTextModel = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14"),
            device: str = "cuda"
            ):
        
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        
        self.device = device
        
        self.text_encoder.to(self.device)
    
    def encode_text(self, text: Union[str, List[str]]) -> torch.Tensor:
        # Convert single string to list
        if isinstance(text, str):
            text = [text]
            
        # Tokenize and move to device
        tokens = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        tokens = {k: v.to(self.device) for k, v in tokens.items()}
        
        # Get text embeddings
        text_embedding = self.text_encoder(**tokens).last_hidden_state
        
        return text_embedding

class CLIPVisionProcessor:
    def __init__(self,
            vision_processor: CLIPProcessor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14"),
            vision_encoder: CLIPVisionModel = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14"),
            device: str = "cuda"
            ):
        
        self.vision_processor = vision_processor
        self.vision_encoder = vision_encoder
        
        self.device = device
        
        self.vision_encoder.to(self.device)
    
    def encode_image(self, image: Union[Image.Image, List[Image.Image]]) -> torch.Tensor:
        # Convert single image to list
        if isinstance(image, Image.Image):
            image = [image]
            
        # Process images and move to device
        pixel_values = self.vision_processor(images=image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)
        
        # Get image embeddings
        image_embedding = self.vision_encoder(pixel_values).last_hidden_state
        
        return image_embedding

class CLIPProcessor:
    def __init__(self, device: str = "cuda"):
        self.text_processor = CLIPTextProcessor(device=device)
        self.vision_processor = CLIPVisionProcessor(device=device)
        
    def encode_text(self, text: Union[str, List[str]]) -> torch.Tensor:
        return self.text_processor.encode_text(text)
    
    def encode_image(self, image: Union[Image.Image, List[Image.Image]]) -> torch.Tensor:
        return self.vision_processor.encode_image(image)
    
    def encode_pairs(self, 
            text: Union[str, List[str]], 
            image: Union[Image.Image, List[Image.Image]]
            ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode both text and image pairs
        Returns tuple of (text_embeddings, image_embeddings)
        """
        text_embeddings = self.encode_text(text)
        image_embeddings = self.encode_image(image)
        return text_embeddings, image_embeddings