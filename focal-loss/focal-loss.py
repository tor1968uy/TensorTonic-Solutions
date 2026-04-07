import numpy as np

def focal_loss(p, y, gamma=2.0):
    """
    Compute Focal Loss for binary classification.
    """
    # Convert to numpy arrays
    p = np.asarray(p)
    y = np.asarray(y)
    
    # Clip probabilities to prevent log(0) or log(1) which leads to nan/inf
    epsilon = 1e-15
    p = np.clip(p, epsilon, 1.0 - epsilon)
    
    # Term 1: for positive class (y=1)
    # loss = -(1-p)^gamma * log(p)
    term1 = (1 - p)**gamma * y * np.log(p)
    
    # Term 2: for negative class (y=0)
    # loss = -p^gamma * log(1-p)
    term2 = p**gamma * (1 - y) * np.log(1 - p)
    
    # Focal loss is the negative mean of the combined terms
    return -np.mean(term1 + term2)