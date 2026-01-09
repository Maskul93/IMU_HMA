import numpy as np

def unit_vector(v):
    '''Normalizes a vector over its norm. It works for both (N x 1) and (N x m) vectors.'''
    v_n = np.zeros_like(v)
    if (len(v) > 1) & (v.ndim > 1):
        for i in range(len(v)):
            v_n[i] = v[i] / np.linalg.norm(v[i])
    
    elif (len(v) > 1) & (v.ndim == 1):
        v_n = v / np.linalg.norm(v)
    
    return v_n