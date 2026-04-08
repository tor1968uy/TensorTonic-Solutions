import numpy as np

def huber_loss(y_true, y_pred, delta=1.0):
    """
    Compute Huber Loss for regression.
    Returns: float (the mean loss across all samples)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # 1. Calculate the raw error
    error = y_true - y_pred
    abs_error = np.abs(error)
    
    # 2. Define the two components of the loss
    # Case 1: Quadratic loss for small errors (|e| <= delta)
    quadratic_loss = 0.5 * (error ** 2)
    
    # Case 2: Linear loss for large errors (|e| > delta)
    # Formula: delta * (|e| - 0.5 * delta)
    linear_loss = delta * (abs_error - 0.5 * delta)
    
    # 3. Use np.where to choose the loss based on the condition |e| <= delta
    sample_losses = np.where(abs_error <= delta, quadratic_loss, linear_loss)
    
    # 4. Return the mean loss across all samples
    return float(np.mean(sample_losses))