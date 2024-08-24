import numpy as np

class Scores:
    
    def __init__(self):
        self.scores = []
        self.best_score = 0.0
    
    def add_score(self, score):
        self.scores.append(score)

        if score > self.best_score:
            self.best_score = score
    
    def get_mean_score(self, last_n: int = 10):
        return np.mean(self.scores[-last_n:])