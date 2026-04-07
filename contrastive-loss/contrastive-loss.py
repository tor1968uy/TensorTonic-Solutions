import numpy as np

def contrastive_loss(a, b, y, margin=1.0, reduction="mean") -> float:
    """
    Compute Contrastive Loss for Siamese pairs.
    a, b: arrays of shape (N, D) or (D,)
    y: array of shape (N,) where 1=similar, 0=dissimilar
    margin: float > 0
    reduction: "mean" or "sum"
    """
    # Convert to numpy arrays
    a = np.asarray(a)
    b = np.asarray(b)
    y = np.asarray(y)
    
    # Ensure a and b have at least 2 dimensions for vectorized norm calculation
    # if input is (D,), it becomes (1, D)
    if a.ndim == 1: a = a[np.newaxis, :]
    if b.ndim == 1: b = b[np.newaxis, :]
    
    # Calculate Euclidean distance d = sqrt(sum((a-b)^2))
    # axis=1 computes the norm across the feature dimension D
    distances = np.linalg.norm(a - b, axis=1)
    
    # Loss for similar pairs (y=1): d^2
    pos_loss = y * (distances ** 2)
    
    # Loss for dissimilar pairs (y=0): max(0, margin - d)^2
    # We use np.maximum to ensure we only penalize if distance < margin
    neg_loss = (1 - y) * (np.maximum(0, margin - distances) ** 2)
    
    # Combine losses
    total_loss_array = pos_loss + neg_loss
    
    # Apply reduction
    if reduction == "sum":
        return float(np.sum(total_loss_array))
    else:
        return float(np.mean(total_loss_array))