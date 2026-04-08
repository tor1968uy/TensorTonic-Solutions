import numpy as np

def hinge_loss(y_true, y_score, margin=1.0, reduction="mean") -> float:
    """
    Compute the hinge loss for binary SVM classification.
    y_true: 1D array of {-1, +1}
    y_score: 1D array of real-valued model scores
    Returns: float (mean or sum of the losses)
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    
    # 1. Compute element-wise product y_i * s_i
    # Positive product means the sign matches (correct classification)
    signed_scores = y_true * y_score
    
    # 2. Compute individual losses: max(0, m - y_i * s_i)
    # If the signed score is greater than the margin, the loss is 0.
    losses = np.maximum(0, margin - signed_scores)
    
    # 3. Apply reduction
    if reduction == "mean":
        return float(np.mean(losses))
    elif reduction == "sum":
        return float(np.sum(losses))
    else:
        raise ValueError("Reduction must be either 'mean' or 'sum'")