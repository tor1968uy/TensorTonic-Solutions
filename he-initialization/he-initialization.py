import math

def he_initialization(W, fan_in):
    """
    Scale raw uniform [0, 1] weights to He uniform range.
    Returns: A list of lists of floats.
    """
    # 1. Compute the He uniform bound (limit)
    # limit = sqrt(6 / fan_in)
    limit = math.sqrt(6.0 / fan_in)
    
    scaled_W = []
    
    for row in W:
        new_row = []
        for weight in row:
            # 2. Map weight from [0, 1] to [-limit, limit]
            # Formula: raw * (range_width) + start_point
            # Range width = 2 * limit; Start point = -limit
            scaled_weight = weight * (2 * limit) - limit
            new_row.append(float(scaled_weight))
        scaled_W.append(new_row)
            
    return scaled_W