import numpy as np

def matrix_trace(A):
    """
    Compute the trace of a square matrix (sum of diagonal elements).
    Returns: scalar (int or float)
    """
    # 1. Ensure input is a NumPy array
    A = np.asarray(A)
    
    # 2. Get the dimension of the square matrix (N x N)
    n = A.shape[0]
    
    # 3. Accumulate the sum of diagonal elements A[i, i]
    trace_sum = 0
    for i in range(n):
        trace_sum += A[i, i]
        
    return trace_sum