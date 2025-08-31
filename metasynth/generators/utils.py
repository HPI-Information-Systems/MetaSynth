import numpy as np


class NormalSampler:
    def __init__(self):
        pass

    def fit(self, data):
        self.mean = np.mean(data)
        self.std = np.std(data)

        return self

    def __call__(self, n):
        return np.random.normal(self.mean, self.std, n)

class CatigoricalSampler:
    def __init__(self):
        pass

    def fit(self, data):
        self.values = np.unique(data)
        self.probs = np.unique(data, return_counts=True)[1] / len(data)

        return self

    def __call__(self, n):
        return np.random.choice(self.values, n, p=self.probs)
    
class UniformSampler:
    def __init__(self):
        pass

    def fit(self, data):
        self.values = np.unique(data)
        self.probs = np.ones(len(self.values)) / len(self.values)

        return self

    def __call__(self, n):
        return np.random.choice(self.values, n, p=self.probs)