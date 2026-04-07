import math

def label_smoothing_loss(predictions, target, epsilon):
    """
    Compute cross-entropy loss with label smoothing.
    Returns a single float.
    """
    K = len(predictions)
    smoothed_loss = 0.0
    
    # 1. Iterate through each class to build the smoothed distribution and compute loss
    for i in range(K):
        # 2. Determine the smoothed target value q_i
        # If it's the target class: (1 - epsilon) + (epsilon / K)
        # If it's any other class: (epsilon / K)
        if i == target:
            q_i = (1.0 - epsilon) + (epsilon / K)
        else:
            q_i = epsilon / K
            
        # 3. Accumulate the cross-entropy: -sum(q_i * ln(p_i))
        # Note: We use math.log for the natural logarithm
        smoothed_loss -= q_i * math.log(predictions[i])
        
    return float(smoothed_loss)