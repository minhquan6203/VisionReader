import string
import torch
import re
from typing import List
import unicodedata
from unidecode import unidecode
import random
import py_vncorenlp

def countTrainableParameters(model) -> int:
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return num_params

def countParameters(model) -> int:
    num_params = sum(p.numel() for p in model.parameters())
    return num_params

def normalize_text(text):
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = text.lower().strip()
    return text

def preprocess_sentence(sentence: str):
    sentence = sentence.lower()
    sentence = unicodedata.normalize('NFC', sentence)
    sentence = re.sub(r"[“”]", "\"", sentence)
    sentence = re.sub(r"!", " ! ", sentence)
    sentence = re.sub(r"\?", " ? ", sentence)
    sentence = re.sub(r":", " : ", sentence)
    sentence = re.sub(r";", " ; ", sentence)
    sentence = re.sub(r",", " , ", sentence)
    sentence = re.sub(r"\"", " \" ", sentence)
    sentence = re.sub(r"'", " ' ", sentence)
    sentence = re.sub(r"\(", " ( ", sentence)
    sentence = re.sub(r"\[", " [ ", sentence)
    sentence = re.sub(r"\)", " ) ", sentence)
    sentence = re.sub(r"\]", " ] ", sentence)
    sentence = re.sub(r"/", " / ", sentence)
    sentence = re.sub(r"\.", " . ", sentence)
    sentence = re.sub(r"-", " - ", sentence)
    sentence = re.sub(r"\$", " $ ", sentence)
    sentence = re.sub(r"\&", " & ", sentence)
    sentence = re.sub(r"\*", " * ", sentence)
    return sentence

def remove_vietnamese_accents(sentence, ratio=0.5):
    output = ''
    for char in sentence:
        if char in 'áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộớờởỡợúùủũụưứừửữựýỳỷỹỵ':
            if random.random() < ratio:
                output += unidecode(char)
            else:
                output += char
        else:
            output += char
    return output

def word_segmentation(sentence):
    py_vncorenlp.download_model(save_dir='./')
    annotator = py_vncorenlp.VnCoreNLP(save_dir='./', annotators=["wseg"])
    sentence = annotator.word_segment(sentence)
    return sentence