import numpy as np


def type2idx(Data_c,Type_c):
    n_samples=len(Data_c)
    target = np.empty((n_samples,), dtype=np.int)
    for idx in range(n_samples):
        if Data_c[idx] in Type_c:
            target[idx]=Type_c.index(Data_c[idx])
        else:
            target[idx] = -1
    return target