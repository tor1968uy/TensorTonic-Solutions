import numpy as np

def triplet_loss(anchor, positive, negative, margin=1.0):
    """
    Compute Triplet Loss for embedding ranking.
    Returns the mean loss across the batch as a float.
    """
    # 1. Convert to numpy arrays
    a = np.asarray(anchor, dtype=float)
    p = np.asarray(positive, dtype=float)
    n = np.asarray(negative, dtype=float)
    
    # 2. Handle shape (D,) by promoting to (1, D) for consistency
    if a.ndim == 1: a = a[np.newaxis, :]
    if p.ndim == 1: p = p[np.newaxis, :]
    if n.ndim == 1: n = n[np.newaxis, :]
    
    # 3. Compute squared Euclidean distances: ||x - y||^2
    # axis=1 computes the sum across the embedding dimensions
    dist_ap = np.sum((a - p)**2, axis=1)
    dist_an = np.sum((a - n)**2, axis=1)
    
    # 4. Apply the Triplet Loss formula: max(0, d(a,p) - d(a,n) + m)
    # np.maximum(0, ...) performs element-wise max
    losses = np.maximum(0, dist_ap - dist_an + margin)
    
    # 5. Return the mean loss across all triplets in the batch
    return float(np.mean(losses))