from typing import Dict, Tuple, List
import numpy as np


class Exact_Match:
    def compute_score(self, y_true, y_pred):
        if y_true==y_pred:
            return 1
        else:
            return 0
