from model.visionreader_bart_vqa import Bart_VQA_Model
from model.visionreader_t5_vqa import T5_VQA_Model

def build_model(config):
    if config['model']['type_model']=='visionreader_t5':
        return T5_VQA_Model(config)
    if config['model']['type_model']=='visionreader_bart':
        return Bart_VQA_Model(config)
    