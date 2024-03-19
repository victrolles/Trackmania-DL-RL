import numpy as np
import torch

# value = torch.tensor([[-0.0770,  0.0902,  0.0253,  0.0755,  0.0763]])
liste = [[1, 2, 3], [1, 2, 3], [1, 2, None]]
a, b, c = zip(*liste)
a = torch.FloatTensor(a)
b = torch.FloatTensor(b)
c = torch.FloatTensor(c)
print(a)
print(b)
print(c)