import torch
from torch import nn
import os
from typing import List
from typing import List, Dict,Any
import numpy as np
import scipy.spatial.distance as distance

class VisionOcrEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()      
        self.linear_det_features = nn.Linear(config.d_det, config.d_model)
        self.linear_rec_features = nn.Linear(config.d_rec, config.d_model)
        self.linear_boxes = nn.Linear(4,config.d_model)

        self.layer_norm_det = nn.LayerNorm(config.d_model)
        self.layer_norm_rec = nn.LayerNorm(config.d_model)
        self.layer_norm_boxes = nn.LayerNorm(config.d_model)

        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(0.1)
        self.cuda_device=config.cuda_device
        self.device = torch.device(f'{self.cuda_device}' if torch.cuda.is_available() else 'cpu')

    def forward(self,ocr_info):
        det_features = torch.stack([det["det_features"] for det in ocr_info]).to(self.device)
        rec_features = torch.stack([rec["rec_features"] for rec in ocr_info]).to(self.device)
        boxes = torch.stack([box["boxes"] for box in ocr_info]).to(self.device)
        
        det_features=self.linear_det_features(det_features)
        rec_features=self.linear_rec_features(rec_features)
        boxes = self.linear_boxes(boxes)
        
        ocr_features = self.layer_norm_det(det_features) + self.layer_norm_rec(rec_features) + self.layer_norm_boxes(boxes)
        ocr_features = self.dropout(self.gelu(ocr_features))
        return ocr_features