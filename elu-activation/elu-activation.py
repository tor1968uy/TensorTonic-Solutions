import math

def elu(x, alpha):
    """
    Apply ELU activation to each element.
    x: List of numbers
    alpha: Float controlling the saturation for negative values
    Returns: List of floats
    """
    output = []
    
    for val in x:
        # 1. Apply identity for positive values
        if val > 0:
            output.append(float(val))
        # 2. Apply alpha * (exp(val) - 1) for negative or zero values
        else:
            # Note: At val = 0, exp(0) is 1, so 1 - 1 = 0.
            transformed = alpha * (math.exp(val) - 1)
            output.append(float(transformed))
            
    return output