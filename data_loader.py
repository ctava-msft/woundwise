import json
import os
from typing import Dict, List, Optional, Tuple
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoProcessor

class WoundWiseDataset(Dataset):
    """Dataset class for WoundWise visual question answering."""
    
    def __init__(
        self,
        data_path: str,
        images_path: str,
        split: str = "train",
        language: str = "en",
        max_length: int = 512,
        processor=None,
        tokenizer=None
    ):
        self.data_path = data_path
        self.images_path = images_path
        self.split = split
        self.language = language
        self.max_length = max_length
        self.processor = processor
        self.tokenizer = tokenizer
        
        # Load data
        self.data = self._load_data()
        
    def _load_data(self) -> List[Dict]:
        """Load dataset from JSON file."""
        file_path = os.path.join(self.data_path, f"mediqa-wv-{self.split}.json")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict:
        item = self.data[idx]
        
        # Get query text
        query_title = item.get(f'query_title_{self.language}', '')
        query_content = item.get(f'query_content_{self.language}', '')
        query_text = f"{query_title} {query_content}".strip()
        
        # Load images
        images = []
        for image_id in item['image_ids']:
            image_path = os.path.join(self.images_path, image_id)
            if os.path.exists(image_path):
                image = Image.open(image_path).convert('RGB')
                images.append(image)
        
        # Get response(s)
        responses = []
        for response in item['responses']:
            response_text = response.get(f'content_{self.language}', '')
            if response_text:
                responses.append(response_text)
        
        # Get metadata
        metadata = {
            'anatomic_locations': item.get('anatomic_locations', []),
            'wound_type': item.get('wound_type', ''),
            'wound_thickness': item.get('wound_thickness', ''),
            'tissue_color': item.get('tissue_color', ''),
            'drainage_amount': item.get('drainage_amount', ''),
            'drainage_type': item.get('drainage_type', ''),
            'infection': item.get('infection', '')
        }
        
        return {
            'encounter_id': item['encounter_id'],
            'query_text': query_text,
            'images': images,
            'responses': responses,
            'metadata': metadata
        }
    
    def collate_fn(self, batch: List[Dict]) -> Dict:
        """Custom collate function for batching."""
        encounter_ids = [item['encounter_id'] for item in batch]
        query_texts = [item['query_text'] for item in batch]
        all_images = [item['images'] for item in batch]
        all_responses = [item['responses'] for item in batch]
        metadata = [item['metadata'] for item in batch]
        
        # Process text
        if self.tokenizer:
            text_inputs = self.tokenizer(
                query_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
        else:
            text_inputs = query_texts
        
        # Process images
        processed_images = []
        if self.processor and all_images:
            for images in all_images:
                if images:
                    # Take first image if multiple
                    image_inputs = self.processor(images[0], return_tensors="pt")
                    processed_images.append(image_inputs)
        
        return {
            'encounter_ids': encounter_ids,
            'text_inputs': text_inputs,
            'images': processed_images if processed_images else all_images,
            'responses': all_responses,
            'metadata': metadata
        }

def create_data_loaders(
    data_path: str,
    images_path: str,
    language: str = "en",
    batch_size: int = 8,
    processor=None,
    tokenizer=None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test data loaders."""
    
    datasets = {}
    for split in ['train', 'valid', 'test']:
        datasets[split] = WoundWiseDataset(
            data_path=data_path,
            images_path=os.path.join(images_path, f"images_{split}"),
            split=split,
            language=language,
            processor=processor,
            tokenizer=tokenizer
        )
    
    data_loaders = {}
    for split, dataset in datasets.items():
        shuffle = split == 'train'
        data_loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=dataset.collate_fn,
            num_workers=4
        )
    
    return data_loaders['train'], data_loaders['valid'], data_loaders['test']
