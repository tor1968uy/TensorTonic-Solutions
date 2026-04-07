import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    Returns: float entropy in bits.
    """
    y = np.asarray(y)
    n_samples = len(y)
    
    # 1. Handle empty node case
    if n_samples == 0:
        return 0.0
    
    # 2. Get class frequencies
    _, counts = np.unique(y, return_counts=True)
    
    # 3. Calculate probabilities p_i = count_i / total
    probs = counts / n_samples
    
    # 4. Compute Entropy: H(S) = -sum(p_i * log2(p_i))
    # Since we only have classes that exist (counts > 0), 
    # we don't need to worry about log2(0).
    entropy = -np.sum(probs * np.log2(probs))
    
    return float(entropy)