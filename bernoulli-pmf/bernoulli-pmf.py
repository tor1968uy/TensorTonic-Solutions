import numpy as np

def bernoulli_pmf_and_moments(x, p):
    """
    Compute Bernoulli PMF and distribution moments.
    Returns: (pmf_array, mean, variance)
    """
    # 1. Convert input to NumPy array
    x = np.asarray(x)
    
    # 2. Compute PMF vectorized: 
    # If x=1, prob is p. If x=0, prob is (1-p).
    pmf = np.where(x == 1, p, 1 - p)
    
    # 3. Compute Mean: E[X] = p
    mean = float(p)
    
    # 4. Compute Variance: Var(X) = p(1 - p)
    variance = float(p * (1 - p))
    
    return pmf, mean, variance