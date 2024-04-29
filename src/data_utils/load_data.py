from typing import List, Dict
from torch.utils.data import DataLoader,Dataset
import json
import os
from utils.utils import preprocess_sentence, remove_vietnamese_accents, word_segmentation

class VQA_dataset(Dataset):
    def __init__(self, annotation_path, image_path, remove_accents_rate=0, use_word_seg=False, with_answer=True):
        with open(annotation_path, 'r') as file:
            json_data = json.load(file)
        
        self.annotations = self.load_annotations(json_data, image_path, remove_accents_rate, use_word_seg, with_answer)
    
    def load_annotations(self, json_data, image_path, remove_accents_rate, use_word_seg, with_answer) -> List[Dict]:
        annotations = []
        if with_answer:
            for ann in json_data["annotations"]:
                # if ann['QA-type']==1: # 1 stand for non-text, 0 mean text 
                    question = ann["question"].replace('?','')
                    if use_word_seg:
                        question = word_segmentation(question)
                        question = ' '.join(question)
                    answer = preprocess_sentence(ann['answers'][0])
                    question = preprocess_sentence(question)
                    question = remove_vietnamese_accents(question,remove_accents_rate)
                    annotation = {
                        "id": ann['id'],
                        "image_id": os.path.join(image_path,str(ann['image_id'])),
                        "question": question,
                        "answer": answer,
                    }
                    annotations.append(annotation)
        else:
            for ann in json_data["annotations"]:
                # if ann['QA-type']==1: # 1 stand for non-text, 0 mean text 
                    question = ann["question"].replace('?','')
                    if use_word_seg:
                        question = word_segmentation(question)
                        question = ' '.join(question)
                    answer = preprocess_sentence(ann['answers'][0])
                    question = preprocess_sentence(question)
                    question = remove_vietnamese_accents(question,remove_accents_rate)
                    annotation = {
                        "id": ann['id'],
                        "image_id": os.path.join(image_path,str(ann['image_id'])),
                        "question": question,
                    }
                    annotations.append(annotation)
        return annotations
    def __getitem__(self, index):
        item = self.annotations[index]
        return item
    def __len__(self) -> int:
        return len(self.annotations)

class Load_Data:
    def __init__(self, config: Dict):
        self.remove_accents_rate = config['text_embedding']['remove_accents_rate']
        self.use_word_seg = config['text_embedding']['use_word_seg']
        self.num_worker = config['data']['num_worker']

        self.train_annotations = config['data']['train_dataset']
        self.train_images = config['data']['images_train_folder']

        self.val_annotations=config["data"]["val_dataset"]
        self.val_images=config['data']['images_val_folder']

        self.test_annotations = config['infer']['test_dataset']
        self.test_images = config['infer']['images_test_folder']

        self.train_batch=config['train']['per_device_train_batch_size']
        self.valid_batch=config['train']['per_device_valid_batch_size']
        self.test_batch=config['infer']['per_device_eval_batch_size']

    def load_train_dev(self):
        train_set=VQA_dataset(self.train_annotations,self.train_images,self.remove_accents_rate,self.use_word_seg)
        val_set=VQA_dataset(self.val_annotations,self.val_images,self.remove_accents_rate,self.use_word_seg)
        train_loader = DataLoader(train_set, batch_size=self.train_batch, num_workers=self.num_worker,shuffle=True)
        val_loader = DataLoader(val_set, batch_size=self.valid_batch, num_workers=self.num_worker,shuffle=True)
        return train_loader, val_loader

    def load_test(self,with_answer):
        test_set=VQA_dataset(self.test_annotations,self.test_images,self.remove_accents_rate,self.use_word_seg,with_answer)
        test_loader = DataLoader(test_set, batch_size=self.test_batch, num_workers=self.num_worker,shuffle=False)
        return test_loader