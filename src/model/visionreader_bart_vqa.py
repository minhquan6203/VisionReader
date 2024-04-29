from typing import List, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from text_module.visionreader_bart_encoding import Bart_Encode_Feature, Bart_Embedding
from vision_module.vision_obj_encoding import  Vision_Encode_Obj_Feature
from vision_module.vision_ocr_encoding import Vision_Encode_Ocr_Feature
from vision_module.vision_pixel_embedding import Vision_Embedding, Vision_Embedding_Extracted
from transformers import AutoConfig, AutoTokenizer

class Bart_VQA_Model(nn.Module):
    def __init__(self,config: Dict):
     
        super(Bart_VQA_Model, self).__init__()
        self.text_encoder = Bart_Encode_Feature(config)
        if config['vision_embedding']['already_extracted']:
            self.vision_embedding = Vision_Embedding_Extracted(config)
        else:
            self.vision_embedding = Vision_Embedding(config)
        self.vision_encoder_ocr = Vision_Encode_Ocr_Feature(config)
        self.vision_encoder_obj = Vision_Encode_Obj_Feature(config)

        self.cuda_device=config['train']['cuda_device']
        self.device = torch.device(f'{self.cuda_device}' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(config['text_embedding']['text_encoder'])
        self.embedding = Bart_Embedding(config)
        self.with_image = config['train']['with_image'] 
        self.generator_args ={
            'max_length': config['generator_args']['max_length'],
            'min_length': config['generator_args']['min_length'],
            'num_beams': config['generator_args']['num_beams'],
            'length_penalty': config['generator_args']['length_penalty'],
            'no_repeat_ngram_size': config['generator_args']['no_repeat_ngram_size'],
            'early_stopping': config['generator_args']['early_stopping'],
        }
    def forward(self, questions: List[str], images: List[str], labels: List[str] = None):
        if self.with_image:
            image_features = self.vision_embedding(images)
            ocr_info = self.vision_encoder_ocr(images)
            obj_info = self.vision_encoder_obj(images)
            ocr_obj_list=[]
            for ocr,obj in zip(ocr_info,obj_info):
                ocr_obj_list.append(f"{ocr['texts']} {obj['object_list']}".strip())
            inputs = self.text_encoder(questions,ocr_obj_list,labels)
            inputs.update({'image_features':image_features,
                            'ocr_info': ocr_info,
                            'obj_info':obj_info})
        else:
            inputs = self.text_encoder(questions,None,labels)

        if labels is not None:
            outputs = self.embedding(**inputs)
            return outputs.logits, outputs.loss
        else:
            pred_ids=self.embedding.generate(**inputs,**self.generator_args)
            pred_tokens=self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
            return pred_tokens