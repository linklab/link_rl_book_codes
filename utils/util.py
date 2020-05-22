import numpy as np

def softmax(x):
    B = np.exp(x - np.max(x))
    C = np.sum(B)
    return B/C