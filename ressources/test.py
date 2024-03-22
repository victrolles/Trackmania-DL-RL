import numpy as np
import torch
from collections import namedtuple
from torch.nn.functional import softmax, one_hot, log_softmax
from torch import argmax


value = torch.tensor([[-0.0770,  0.0902,  0.0253,  0.0755,  0.0763]])
print(value)
probabilities = softmax(value, dim=-1)
print(probabilities)
log_probabilities = log_softmax(probabilities, dim=-1)
print(log_probabilities)
best = argmax(probabilities, dim=1)
print(best)
random_dist = torch.distributions.Categorical(probabilities)
print(random_dist)
action = random_dist.sample()
print(action)
log_probabilities = random_dist.log_prob(action)
print(log_probabilities)