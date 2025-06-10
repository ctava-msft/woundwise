import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, AutoProcessor, AutoModel,
    BlipProcessor, BlipForConditionalGeneration,
    T5ForConditionalGeneration, T5Tokenizer
)
from typing import Dict, List, Optional, Tuple

class WoundWiseVQAModel(nn.Module):
    """Visual Question Answering model for wound care."""
    
    def __init__(
        self,
        model_name: str = "Salesforce/blip-image-captioning-base",
        language: str = "en",
        max_length: int = 512
    ):
        super().__init__()
        self.model_name = model_name
        self.language = language
        self.max_length = max_length
        
        # Initialize vision-language model
        if "blip" in model_name.lower():
            self.processor = BlipProcessor.from_pretrained(model_name)
            self.model = BlipForConditionalGeneration.from_pretrained(model_name)
        else:
            # Fallback to generic multimodal model
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
        
        # Text tokenizer for processing responses
        if language == "zh":
            self.text_tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
        else:
            self.text_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    def forward(
        self,
        images: List[torch.Tensor],
        query_texts: List[str],
        target_texts: Optional[List[str]] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for VQA model."""
        
        batch_size = len(query_texts)
        device = next(self.parameters()).device
        
        # Process inputs
        processed_inputs = []
        for i in range(batch_size):
            if images[i] is not None and len(images[i]) > 0:
                # Use first image if multiple images
                image = images[i][0] if isinstance(images[i], list) else images[i]
                inputs = self.processor(
                    image,
                    query_texts[i],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length
                )
                processed_inputs.append(inputs)
        
        if not processed_inputs:
            return {"loss": torch.tensor(0.0, device=device)}
        
        # Batch processing
        pixel_values = torch.cat([inp['pixel_values'] for inp in processed_inputs]).to(device)
        input_ids = torch.cat([inp['input_ids'] for inp in processed_inputs]).to(device)
        attention_mask = torch.cat([inp['attention_mask'] for inp in processed_inputs]).to(device)
        
        if target_texts is not None:
            # Training mode - compute loss
            target_encodings = self.text_tokenizer(
                target_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            ).to(device)
            
            outputs = self.model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=target_encodings['input_ids']
            )
            
            return {
                "loss": outputs.loss,
                "logits": outputs.logits
            }
        else:
            # Inference mode - generate response
            outputs = self.model.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=self.max_length,
                num_beams=4,
                early_stopping=True,
                do_sample=True,
                temperature=0.7
            )
            
            return {"generated_ids": outputs}
    
    def generate_response(
        self,
        image: torch.Tensor,
        query_text: str
    ) -> str:
        """Generate a response for a single image-query pair."""
        
        self.eval()
        with torch.no_grad():
            outputs = self.forward([image], [query_text])
            generated_ids = outputs["generated_ids"]
            
            # Decode generated text
            response = self.text_tokenizer.decode(
                generated_ids[0],
                skip_special_tokens=True
            )
            
        return response

class MultimodalEncoder(nn.Module):
    """Custom multimodal encoder for wound care VQA."""
    
    def __init__(
        self,
        vision_model_name: str = "microsoft/resnet-50",
        text_model_name: str = "bert-base-uncased",
        hidden_dim: int = 768,
        num_layers: int = 2
    ):
        super().__init__()
        
        # Vision encoder
        self.vision_encoder = AutoModel.from_pretrained(vision_model_name)
        self.vision_projection = nn.Linear(
            self.vision_encoder.config.hidden_size,
            hidden_dim
        )
        
        # Text encoder
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        self.text_projection = nn.Linear(
            self.text_encoder.config.hidden_size,
            hidden_dim
        )
        
        # Fusion layers
        self.fusion_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                batch_first=True
            )
            for _ in range(num_layers)
        ])
        
        # Response generation head
        self.response_head = nn.Linear(hidden_dim, text_model_name.vocab_size)
        
    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass through multimodal encoder."""
        
        # Encode vision
        vision_outputs = self.vision_encoder(pixel_values)
        vision_features = self.vision_projection(vision_outputs.last_hidden_state)
        
        # Encode text
        text_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        text_features = self.text_projection(text_outputs.last_hidden_state)
        
        # Concatenate features
        combined_features = torch.cat([vision_features, text_features], dim=1)
        
        # Apply fusion layers
        for layer in self.fusion_layers:
            combined_features = layer(combined_features)
        
        # Generate response logits
        response_logits = self.response_head(combined_features)
        
        return response_logits
