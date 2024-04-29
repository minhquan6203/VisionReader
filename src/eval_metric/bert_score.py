from evaluate import load
import numpy as np

class Bert_Score:
    def __init__(self):
        self.bertscore = load("bertscore")
    def compute_score(self, y_true, y_pred):
        results = self.bertscore.compute(predictions=y_pred, references=y_true, model_type="bert-base-multilingual-cased")
        return np.mean(results['f1'][0])