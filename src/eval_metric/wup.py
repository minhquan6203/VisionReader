from typing import Dict, Tuple, List
import numpy as np
from nltk.corpus import wordnet

class Wup:

    def get_semantic_field(self,a):
        weight = 1.0
        semantic_field = wordnet.synsets(a,pos=wordnet.NOUN)
        return (semantic_field,weight)

    def get_stem_word(self,a):
        """
        Sometimes answer has form word\d+:wordid.
        If so we return word and downweight
        """
        weight = 1.0
        return (a,weight)
    def compute_score(self, a: str, b: str, similarity_threshold: float = 0.9):
        """
        Returns Wu-Palmer similarity score.
        More specifically, it computes:
            max_{x \in interp(a)} max_{y \in interp(b)} wup(x,y)
            where interp is a 'interpretation field'
        """
        global_weight=1.0

        (a,global_weight_a)=self.get_stem_word(a)
        (b,global_weight_b)=self.get_stem_word(b)
        global_weight = min(global_weight_a,global_weight_b)

        if a==b:
            # they are the same
            return 1.0*global_weight

        if a==[] or b==[]:
            return 0

        interp_a,weight_a = self.get_semantic_field(a) 
        interp_b,weight_b = self.get_semantic_field(b)

        if interp_a == [] or interp_b == []:
            return 0

        # we take the most optimistic interpretation
        global_max=0.0
        for x in interp_a:
            for y in interp_b:
                local_score=x.wup_similarity(y)
                if local_score > global_max:
                    global_max=local_score

        # we need to use the semantic fields and therefore we downweight
        # unless the score is high which indicates both are synonyms
        if global_max < similarity_threshold:
            interp_weight = 0.1
        else:
            interp_weight = 1.0

        final_score=global_max*weight_a*weight_b*interp_weight*global_weight
        return final_score
    