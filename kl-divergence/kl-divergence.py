import numpy as np

def kl_divergence(p, q, eps=1e-12):
    """
    Compute KL Divergence D_KL(P || Q).
    p: Reference distribution (array-like)
    q: Approximation distribution (array-like)
    eps: Small value for numerical stability
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    
    # 1. Filter indices where p > 0 to follow the 0 * log(0) = 0 convention
    mask = p > 0
    p_pos = p[mask]
    q_pos = q[mask]
    
    # 2. Add epsilon to q to avoid log(0) and division by zero
    q_stable = q_pos + eps
    
    # 3. Calculate D_KL(P || Q) = sum(P * log(P / Q))
    # We use np.log for the natural logarithm (base e)
    divergence = np.sum(p_pos * np.log(p_pos / q_stable))
    
    return float(divergence)