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
        images: List,
        query_texts: List[str],
        target_texts: Optional[List[str]] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for VQA model."""
        
        batch_size = len(query_texts)
        device = next(self.parameters()).device
        
        # Process inputs individually first
        processed_inputs = []
        valid_indices = []
        
        for i in range(batch_size):
            # Handle different image formats
            image_list = images[i] if i < len(images) else []
            
            if image_list and len(image_list) > 0:
                # Get the first image (PIL Image)
                image = image_list[0]
                
                # Process with the processor
                try:
                    inputs = self.processor(
                        image,
                        query_texts[i],
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=self.max_length
                    )
                    processed_inputs.append(inputs)
                    valid_indices.append(i)
                except Exception as e:
                    print(f"Error processing image {i}: {e}")
                    continue
        
        if not processed_inputs:
            return {"loss": torch.tensor(0.0, device=device)}
        
        # Batch processing with proper padding
        try:
            # Collect all pixel values - these should be the same size
            pixel_values_list = []
            input_ids_list = []
            attention_mask_list = []
            
            for inputs in processed_inputs:
                pixel_values_list.append(inputs['pixel_values'].squeeze(0))
                input_ids_list.append(inputs['input_ids'].squeeze(0))
                attention_mask_list.append(inputs['attention_mask'].squeeze(0))
            
            # Stack pixel values (should be same size)
            pixel_values = torch.stack(pixel_values_list).to(device)
            
            # Pad sequences to same length
            max_seq_len = max(ids.size(0) for ids in input_ids_list)
            
            padded_input_ids = []
            padded_attention_masks = []
            
            for input_ids, attention_mask in zip(input_ids_list, attention_mask_list):
                # Pad to max length
                seq_len = input_ids.size(0)
                if seq_len < max_seq_len:
                    # Pad with tokenizer's pad_token_id
                    pad_token_id = self.processor.tokenizer.pad_token_id or 0
                    padding = torch.full((max_seq_len - seq_len,), pad_token_id, dtype=input_ids.dtype)
                    padded_input_ids.append(torch.cat([input_ids, padding]))
                    
                    # Pad attention mask with zeros
                    att_padding = torch.zeros((max_seq_len - seq_len,), dtype=attention_mask.dtype)
                    padded_attention_masks.append(torch.cat([attention_mask, att_padding]))
                else:
                    padded_input_ids.append(input_ids)
                    padded_attention_masks.append(attention_mask)
            
            input_ids = torch.stack(padded_input_ids).to(device)
            attention_mask = torch.stack(padded_attention_masks).to(device)
            
        except Exception as e:
            print(f"Error batching inputs: {e}")
            print(f"Number of processed inputs: {len(processed_inputs)}")
            for i, inputs in enumerate(processed_inputs):
                print(f"Input {i} shapes:")
                for key, value in inputs.items():
                    if torch.is_tensor(value):
                        print(f"  {key}: {value.shape}")
            return {"loss": torch.tensor(0.0, device=device)}
        
        if target_texts is not None:
            # Training mode - compute loss
            # Only use target texts for samples that have valid images
            valid_targets = [target_texts[i] for i in valid_indices]
            
            target_encodings = self.text_tokenizer(
                valid_targets,
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
        image,
        query_text: str
    ) -> str:
        """Generate a response for a single image-query pair."""
        
        self.eval()
        device = next(self.parameters()).device
        
        with torch.no_grad():
            try:
                # Process single image and text
                inputs = self.processor(
                    image,
                    query_text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length
                )
                
                # Move to device
                for key in inputs:
                    if torch.is_tensor(inputs[key]):
                        inputs[key] = inputs[key].to(device)
                
                # Generate response
                generated_ids = self.model.generate(
                    pixel_values=inputs['pixel_values'],
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_length=self.max_length,
                    num_beams=4,
                    early_stopping=True,
                    do_sample=True,
                    temperature=0.7
                )
                
                # Decode generated text
                response = self.text_tokenizer.decode(
                    generated_ids[0],
                    skip_special_tokens=True
                )
                
                return response
                
            except Exception as e:
                print(f"Error generating response: {e}")
                return "Unable to generate response for this image."

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
