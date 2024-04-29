import torch
from torch import nn
import os
from typing import List
from collections import Counter
from typing import List, Dict,Any
import numpy as np
from utils.utils import preprocess_sentence

class Vision_Encode_Obj_Feature(nn.Module):
    def __init__(self, config: Dict):
        super(Vision_Encode_Obj_Feature,self).__init__()
        self.cuda_device=config['train']['cuda_device']
        self.device = torch.device(f'{self.cuda_device}' if torch.cuda.is_available() else 'cpu')
        self.obj_features_path = config['obj_embedding']['path_obj']
        self.max_bbox = config['obj_embedding']['max_bbox']
        self.d_obj=config['obj_embedding']['d_obj']
        self.d_grid=config['obj_embedding']['d_grid']
        self.use_attr=config['obj_embedding']['use_attr']
         
    def forward(self, images: List[str]):
        obj_info = [self.load_obj_features(image_id) for image_id in images]
        return obj_info
    
    def pad_array(self, array: np.ndarray, max_len: int, value):
        if max_len == 0:
            array= np.zeros((0, array.shape[-1]))
        else:
            pad_value_array = np.zeros((max_len-array.shape[0], array.shape[-1])).fill(value)
            array = np.concatenate([array, pad_value_array], axis=0)
        return array

    def pad_tensor(self, tensor: torch.Tensor, max_len: int, value):
        if max_len == 0:
            tensor = torch.zeros((0, tensor.shape[-1]))
        else:
            pad_value_tensor = torch.zeros((max_len-tensor.shape[0], tensor.shape[-1])).fill_(value)
            tensor = torch.cat([tensor, pad_value_tensor], dim=0)
        return tensor

    def pad_list(self, list: List, max_len: int, value):
        pad_value_list = [value] * (max_len - len(list))
        list.extend(pad_value_list)
        return list

    def get_size_obj(self, image_id: int):
        feature_file = os.path.join(self.obj_features_path, f"{image_id}.npy")
        if os.path.exists(feature_file):
            features = np.load(feature_file, allow_pickle=True)[()]
            w,h=features['width'],features['height']
            return torch.tensor([w,h,w,h])
        else:
            return torch.tensor([1,1,1,1])
        
    def load_obj_features(self, image_id: int) -> Dict[str, Any]:
        image_id = os.path.basename(image_id).split('.')[0]
        feature_file = os.path.join(self.obj_features_path, f"{int(image_id)}.npy")
        if os.path.exists(feature_file):
            features = np.load(feature_file, allow_pickle=True)[()]
            for key, feature in features.items():
                if isinstance(feature, np.ndarray):
                    features[key] = torch.tensor(feature)
    
            if features['region_features'].shape[0] > self.max_bbox:
                region_features=features['region_features'][:self.max_bbox]
                region_boxes=features['region_boxes'][:self.max_bbox]
            else:
                region_features=self.pad_tensor(features['region_features'],self.max_bbox,0.)
                region_boxes=self.pad_tensor(features['region_boxes'],self.max_bbox,0.)
                
            if self.use_attr:  
                if len(features['object_list'])> self.max_bbox:
                    features['object_list']=features['object_list'][:self.max_bbox]
                obj_attr_list=[]
                for i in range(len(features['object_list'])):
                    features['attr_list'][i] = [attr.lower() for attr in features['attr_list'][i] if attr != 'táo']
                    obj_attr_list.append(f"{features['object_list'][i]}: {' '.join(features['attr_list'][i][:4])}")
                features['object_list']='' if self.max_bbox==0 else preprocess_sentence(','.join(obj_attr_list))
            else:
                if len(features['object_list'])> self.max_bbox:
                    features['object_list']=features['object_list'][:self.max_bbox]
                features['object_list']='' if self.max_bbox==0 else preprocess_sentence(' '.join(features['object_list']))
            
            obj_info={
                'region_features': region_features.detach().cpu(),
                'region_boxes': (region_boxes/self.get_size_obj(image_id)).detach().cpu(),
                'grid_features': features['grid_features'].detach().cpu(),
                'grid_boxes': features['grid_boxes'].squeeze(0).detach().cpu(),
                'object_list': features['object_list'],
                'height': features['height'],
                'width': features['width'],
            }
        else:
            print('path not found, model auto padding features')
            region_features=self.pad_tensor(torch.zeros(1,self.d_obj), self.max_bbox, 0.)
            region_boxes=self.pad_tensor(torch.zeros(1,4), self.max_bbox, 0.)
            grid_features=self.pad_tensor(torch.zeros(1,self.d_grid), 49, 0.)
            grid_boxes=self.pad_tensor(torch.zeros(1,4), 49, 0.)
            obj_info={
                'region_features': region_features.detach().cpu(),
                'region_boxes': region_boxes.detach().cpu(),
                'grid_features': grid_features.detach().cpu(),
                'grid_boxes': grid_boxes.detach().cpu(),
                'object_list': '',
                'width': 1,
                'height': 1,

            }
        return obj_info