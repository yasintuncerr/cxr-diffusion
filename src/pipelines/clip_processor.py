import torch
import PIL.Image
from typing import List, Optional, Tuple, Union
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection, CLIPTokenizer, CLIPTextModel
from PIL import Image


class ClipVisionProcessor:
    
    def __init__(
        self,
        device: Union[str, torch.device] = "cuda",
        image_encoder: Optional[CLIPVisionModelWithProjection] = None,
        feature_extractor: Optional[CLIPImageProcessor] = None
    ):
        """
        Initialize the CLIPVisionEmbedder.
        
        Args:
            device: Device to use for processing
            image_encoder: The CLIP Vision model (if None, will be loaded from pretrained)
            feature_extractor: The image processor (if None, will be loaded from pretrained)
        """
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        
        # Load pretrained models if not provided
        if image_encoder is None or feature_extractor is None:
            from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
            
            model_id = "openai/clip-vit-large-patch14"
            if image_encoder is None:
                self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(model_id)
            else:
                self.image_encoder = image_encoder
                
            if feature_extractor is None:
                self.feature_extractor = CLIPImageProcessor.from_pretrained(model_id)
            else:
                self.feature_extractor = feature_extractor
        else:
            self.image_encoder = image_encoder
            self.feature_extractor = feature_extractor
        
        # Move model to device
        self.image_encoder = self.image_encoder.to(self.device)
    
    def encode_image(
        self, 
        image: Union[PIL.Image.Image, List[PIL.Image.Image], torch.Tensor], 
        num_images_per_prompt: int = 1,
        output_hidden_states: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Encode an image or batch of images to generate embeddings.
        
        Args:
            image: Image or batch of images to encode
            num_images_per_prompt: Number of images to generate per prompt
            output_hidden_states: Whether to return hidden states
            
        Returns:
            If output_hidden_states is False:
                Tuple of (image_embeds, uncond_image_embeds)
            If output_hidden_states is True:
                Tuple of (image_enc_hidden_states, uncond_image_enc_hidden_states)
        """
        # Get model dtype
        dtype = next(self.image_encoder.parameters()).dtype

        # Process input image
        if not isinstance(image, torch.Tensor):
            image = self.feature_extractor(image, return_tensors="pt").pixel_values

        # Move to device and set dtype
        image = image.to(device=self.device, dtype=dtype)
        
        # Get embeddings based on requested output type
        if output_hidden_states:
            # Get hidden states for conditional embeddings
            image_enc_output = self.image_encoder(image, output_hidden_states=True)
            image_enc_hidden_states = image_enc_output.hidden_states[-2]
            image_enc_hidden_states = image_enc_hidden_states.repeat_interleave(num_images_per_prompt, dim=0)
            
            # Generate unconditional embeddings (zeros) for classifier-free guidance
            uncond_image_enc_output = self.image_encoder(
                torch.zeros_like(image), output_hidden_states=True
            )
            uncond_image_enc_hidden_states = uncond_image_enc_output.hidden_states[-2]
            uncond_image_enc_hidden_states = uncond_image_enc_hidden_states.repeat_interleave(
                num_images_per_prompt, dim=0
            )
            
            return image_enc_hidden_states, uncond_image_enc_hidden_states
        else:
            # Get standard embeddings
            image_embeds = self.image_encoder(image).image_embeds
            image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)
            
            # Generate unconditional embeddings (zeros) for classifier-free guidance
            uncond_image_embeds = torch.zeros_like(image_embeds)

            return image_embeds, uncond_image_embeds
            
    

class CLIPTextProcessor:
    
    def __init__(
        self,
        device: Union[str, torch.device] = "cuda",
        tokenizer: Optional[CLIPTokenizer] = None,
        text_encoder: Optional[CLIPTextModel] = None
    ):
        """
        Initialize the CLIPTextProcessor.
        
        Args:
            device: Device to use for processing
            tokenizer: CLIP tokenizer (if None, will be loaded from pretrained)
            text_encoder: CLIP text model (if None, will be loaded from pretrained)
        """
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        
        # Load pretrained models if not provided
        model_id = "openai/clip-vit-large-patch14"
        
        if tokenizer is None:
            self.tokenizer = CLIPTokenizer.from_pretrained(model_id)
        else:
            self.tokenizer = tokenizer
            
        if text_encoder is None:
            self.text_encoder = CLIPTextModel.from_pretrained(model_id)
        else:
            self.text_encoder = text_encoder
        
        # Move model to device
        self.text_encoder = self.text_encoder.to(self.device)
    
    def encode_text(self, text: Union[str, List[str]]) -> torch.Tensor:
        """
        Encode text prompts into embeddings.
        
        Args:
            text: Text prompt or list of text prompts to encode
            
        Returns:
            Tensor of text embeddings
        """
        # Convert single string to list
        if isinstance(text, str):
            text = [text]
            
        # Tokenize and move to device
        tokens = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        tokens = {k: v.to(self.device) for k, v in tokens.items()}
        
        # Get text embeddings
        text_embedding = self.text_encoder(**tokens).last_hidden_state
        
        return text_embedding