def exponential_moving_average(values, alpha):
    """
    Compute the exponential moving average of the given values.
    Returns: a list of floats of the same length as values.
    """
    if not values:
        return []
        
    # 1. Initialize EMA list with the first value
    ema_results = [float(values[0])]
    
    # 2. Iterate through the rest of the values starting from index 1
    for i in range(1, len(values)):
        current_x = values[i]
        previous_ema = ema_results[-1]
        
        # 3. Apply the recursive formula:
        # EMA_t = (alpha * current_value) + ((1 - alpha) * previous_ema)
        current_ema = (alpha * current_x) + (1 - alpha) * previous_ema
        
        ema_results.append(float(current_ema))
            
    return ema_results