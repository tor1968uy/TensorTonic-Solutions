import math

def log_loss(y_true, y_pred, eps=1e-15):
    """
    Compute per-sample log loss.
    Returns: A list of floats representing the loss for each sample.
    """
    losses = []
    
    for y, p in zip(y_true, y_pred):
        # 1. Clip probability to avoid log(0) or log(1)
        # p_clipped will be in range [eps, 1 - eps]
        p_clipped = max(eps, min(1 - eps, p))
        
        # 2. Apply the binary cross-entropy formula:
        # Loss = -(y * ln(p) + (1 - y) * ln(1 - p))
        # Note: If y=1, only the first term matters. If y=0, only the second.
        sample_loss = -(y * math.log(p_clipped) + (1 - y) * math.log(1 - p_clipped))
        
        losses.append(sample_loss)
        
    return losses