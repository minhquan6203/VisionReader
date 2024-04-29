import torch
from torch import nn
import os
from PIL import Image
from typing import List
from transformers import AutoFeatureExtractor, AutoModel
from collections import Counter
from typing import List, Dict, Optional,Any
import numpy as np
from vision_module.vision_pixel_encoding import Vision_Encode_Pixel

class Vision_Embedding(nn.Module):
    def __init__(self, config: Dict) -> None:
        super(Vision_Embedding,self).__init__()     
        self.visual_embedding = AutoModel.from_pretrained(config['vision_embedding']['image_encoder'])
        self.visual_encoding = Vision_Encode_Pixel(config)
        for param in self.visual_embedding.parameters():
            param.requires_grad = False

    def forward(self, image_ids: List[str]):
        pixels=self.visual_encoding(image_ids)
        featrues=self.visual_embedding(pixels).last_hidden_state
        return featrues


class Vision_Embedding_Extracted(nn.Module):
    def __init__(self, config: Dict):
        super(Vision_Embedding_Extracted,self).__init__()
        self.cuda_device = config['train']['cuda_device']
        self.device = torch.device(f"{self.cuda_device}" if torch.cuda.is_available() else "cpu")
        self.feature_path = config["vision_embedding"]["feature_path"]
         
    def forward(self, image_ids: List[str]):
        features_list=[]
        for image_id in image_ids:
            image_id = os.path.basename(image_id).split('.')[0]
            feature_file = os.path.join(self.feature_path, f"{int(image_id)}.npy")
            feature = np.load(feature_file, allow_pickle=True)[()]
            features_list.append(torch.tensor(feature['image_feature']))
        features=torch.stack(features_list).to(self.device)
        return features