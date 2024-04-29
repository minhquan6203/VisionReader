from typing import Dict, Tuple, List
import numpy as np
from eval_metric.f1 import F1
# from eval_metric.wup import Wup
from eval_metric.em import Exact_Match
from eval_metric.bert_score import Bert_Score
from eval_metric.cider import CiderScorer
from utils.utils import normalize_text,preprocess_sentence

class ScoreCalculator:
    def __init__(self):
        self.f1_caculate=F1()
        self.em_caculate=Exact_Match()
        # self.Wup_caculate=Wup()
        self.bert_caculate=Bert_Score()
    #F1 score character level
    def f1_char(self,labels: List[str], preds: List[str]) -> float:        
        scores=[]
        for i in range(len(labels)):
            scores.append(self.f1_caculate.compute_score(preprocess_sentence(normalize_text(labels[i])),preprocess_sentence(normalize_text(preds[i]))))
        return np.mean(scores)

    #F1 score token level
    def f1_token(self,labels: List[str], preds: List[str]) -> float:
        scores=[]
        for i in range(len(labels)):
            scores.append(self.f1_caculate.compute_score(preprocess_sentence(normalize_text(labels[i])).split(),preprocess_sentence(normalize_text(preds[i])).split()))
        return np.mean(scores)
    #Excat match score
    def em(self,labels: List[str], preds: List[str]) -> float:
        scores=[]
        for i in range(len(labels)):
            scores.append(self.em_caculate.compute_score(preprocess_sentence(normalize_text(labels[i])),preprocess_sentence(normalize_text(preds[i]))))
        return np.mean(scores)
    #Wup score
    def wup(self,labels: List[str], preds: List[str]) -> float:
        # scores=[]
        # for i in range(len(labels)):
        #     scores.append(self.Wup_caculate.compute_score(preprocess_sentence(normalize_text(labels[i])),preprocess_sentence(normalize_text(preds[i]))))
        # return np.mean(scores)
        return 0
    #Bert score
    def bert_score(self,labels: List[str], preds: List[str]) -> float:
        labels=[preprocess_sentence(normalize_text(label)) for label in labels]
        preds=[preprocess_sentence(normalize_text(pred)) for pred in preds ]
        scores=self.bert_caculate.compute_score(labels,preds)
        return scores
    #Cider score
    def cider_score(self,labels: List[str], preds: List[str]) -> float:
        labels=[[preprocess_sentence(normalize_text(label))] for label in labels]
        preds=[[preprocess_sentence(normalize_text(pred))] for pred in preds ]
        cider_caculate= CiderScorer(labels, test=preds, n=4, sigma=6.)
        scores,_=cider_caculate.compute_score()
        return scores