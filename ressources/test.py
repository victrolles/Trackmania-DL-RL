import numpy as np

liste = [(1, "a"), (2, "b"), (3, "c"), (4, "d"), (5, "e")]

indices = np.random.choice(len(liste), 3, replace=False)
states, actions = zip(*[liste.pop(idx) for idx in indices])
print(f"states : {states}")
print(f"actions : {actions}")
print(f"new list : {liste}")