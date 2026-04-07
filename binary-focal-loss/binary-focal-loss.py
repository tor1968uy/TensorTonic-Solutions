import math

def binary_focal_loss(predictions, targets, alpha, gamma):
    """
    Compute the mean binary focal loss.
    """
    total_loss = 0.0
    n = len(predictions)
    
    for p, y in zip(predictions, targets):
        # 1. Determine p_t: the probability assigned to the true class
        if y == 1:
            p_t = p
        else:
            p_t = 1 - p
            
        # 2. Compute the focal loss for this specific sample
        # FL = -alpha * (1 - p_t)^gamma * ln(p_t)
        # Using math.log for the natural logarithm
        modulating_factor = (1 - p_t) ** gamma
        log_pt = math.log(p_t)
        
        sample_loss = -alpha * modulating_factor * log_pt
        
        # 3. Accumulate the loss
        total_loss += sample_loss
        
    # Return the mean focal loss
    return total_loss / n