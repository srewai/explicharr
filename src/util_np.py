import numpy as np


def jagged_array(x, fill, shape, dtype):
    """-> np.ndarray; with jagged `x` trimmed and filled into `shape`."""
    a = np.full(shape= shape, dtype= dtype, fill_value= fill)
    i, *shape = shape
    if shape:
        x = np.stack([jagged_array(x, fill, shape, dtype) for x in x])
    else:
        x = np.fromiter(x, dtype= dtype)
    i = min(i, len(x))
    a[:i] = x[:i]
    return a


def permute(n, seed= 0):
    """returns a random permutation of the first `n` natural numbers."""
    np.random.seed(0)
    i = np.arange(n)
    np.random.shuffle(i)
    return i
