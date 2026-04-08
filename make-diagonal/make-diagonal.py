import numpy as np

def make_diagonal(v):
    """
    Constructs an n x n diagonal matrix from a 1D vector v.
    Returns: (n, n) NumPy array.
    """
    # Ensure input is a 1D NumPy array
    v = np.asarray(v)
    
    # np.diag(v) returns a square matrix with v on the main diagonal
    return np.diag(v)