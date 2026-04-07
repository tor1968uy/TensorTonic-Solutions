import numpy as np
from scipy.special import comb

def binomial_pmf_cdf(n, p, k):
    """
    Compute Binomial PMF and CDF for k successes in n trials.
    Returns: (pmf, cdf) as a tuple of floats.
    """
    # 1. Function to calculate PMF for a specific i
    def get_pmf(i):
        # Formula: C(n, i) * p^i * (1-p)^(n-i)
        coefficient = comb(n, i)
        return coefficient * (p**i) * ((1 - p)**(n - i))

    # 2. Calculate PMF for exactly k successes
    pmf_k = get_pmf(k)
    
    # 3. Calculate CDF by summing PMFs from 0 up to k
    # P(X <= k) = P(X=0) + P(X=1) + ... + P(X=k)
    cdf_k = sum(get_pmf(i) for i in range(k + 1))
    
    return float(pmf_k), float(cdf_k)