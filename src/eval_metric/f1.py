
from typing import Dict, Tuple, List
import numpy as np

class F1:
  def Precision(self,y_true,y_pred):
    if y_pred is None:
       return 0
    common = set(y_true) & set(y_pred)
    return len(common) / len(set(y_pred))

  def Recall(self,y_true,y_pred):
    common = set(y_true) & set(y_pred)
    return len(common) / len(set(y_true))

  def compute_score(self,y_true,y_pred):
    if len(y_pred) == 0 or len(y_true) == 0:
        return int(y_pred == y_true)

    precision = self.Precision(y_true, y_pred)
    recall = self.Recall(y_true, y_pred)

    if precision == 0 or recall == 0:
        return 0
    f1 = 2*precision*recall / (precision+recall)
    return f1
  

