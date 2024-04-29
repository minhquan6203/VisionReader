import torch
from torch import nn
import os
from PIL import Image
from typing import List
from transformers import AutoFeatureExtractor, AutoModel
from collections import Counter
from typing import List, Dict, Optional,Any
import numpy as np

class Vision_Encode_Pixel(nn.Module):
    def __init__(self, config: Dict):
        super(Vision_Encode_Pixel,self).__init__()
        self.preprocessor = AutoFeatureExtractor.from_pretrained(config["vision_embedding"]["image_encoder"])
        self.cuda_device=config['train']['cuda_device']
        self.device = torch.device(f'{self.cuda_device}' if torch.cuda.is_available() else 'cpu')
         
    def forward(self, images: List[str]):
        processed_images = self.preprocessor(
            images=[
                self.load_image(image_id) for image_id in images
            ],
            return_tensors="pt",
        ).to(self.device)
        return processed_images.pixel_values

    def load_image(self, images):
        for extension in ['jpg', 'png', 'jpeg', 'JPG']:
            image_path = images + "." + extension
            if os.path.exists(image_path):
                image = Image.open(image_path).convert('RGB')
                return image
        raise FileNotFoundError(f"Image not found for {images}")
