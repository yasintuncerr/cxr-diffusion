import torch
from transformers import CLIPTokenizer, CLIPTextModel
from typing import Union, List

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
        # Tek string ise liste haline getir
        if isinstance(text, str):
            text = [text]
            
        # Tokenize ve device'a gönder
        tokens = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        tokens = {k: v.to(self.device) for k, v in tokens.items()}
        
        # Text embeddings al
        text_embedding = self.text_encoder(**tokens).last_hidden_state
        
        return text_embedding