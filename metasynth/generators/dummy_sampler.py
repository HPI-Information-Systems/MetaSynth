from typing import Any, List

import pandas as pd
from metasynth.generators.utils import CatigoricalSampler as CS
from metasynth.generators.utils import UniformSampler as US


class MarginalSampler():

    def __init__(self):
        pass
    
    def fit(self, X):
        
        self.columns = X.columns.tolist()
        self.samplers = []

        for col in self.columns:
            self.samplers.append(CS().fit(X[col]))
   
    def generate(self, count: int):

        data = {col: sampler(count) for col, sampler in zip(self.columns, self.samplers)}

        return pd.DataFrame(data)
        
        
class UniformSampler():

    def __init__(self):
        pass
    
    def fit(self, X):
        
        self.columns = X.columns.tolist()
        self.samplers = []

        for col in self.columns:
            self.samplers.append(US().fit(X[col]))
   
    def generate(self, count: int):

        data = {col: sampler(count) for col, sampler in zip(self.columns, self.samplers)}

        return pd.DataFrame(data)
    