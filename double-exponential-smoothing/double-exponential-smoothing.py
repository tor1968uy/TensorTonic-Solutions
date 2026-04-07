def double_exponential_smoothing(series, alpha, beta):
    """
    Apply Holt's linear trend method.
    Returns: list of level values [l_0, l_1, ..., l_{n-1}].
    """
    n = len(series)
    if n < 2:
        return [float(x) for x in series]

    # 1. Initialize l_0 and b_0
    # Level starts at the first observation
    # Trend starts as the difference between the first two observations
    l_t = float(series[0])
    b_t = float(series[1] - series[0])
    
    # Store the initial level
    levels = [l_t]
    
    # 2. Iterate through the series starting from the second element (index 1)
    for t in range(1, n):
        y_t = series[t]
        
        # Capture previous states for the calculation
        l_prev = l_t
        b_prev = b_t
        
        # 3. Update Level: blends actual y_t with predicted (l_prev + b_prev)
        l_t = alpha * y_t + (1 - alpha) * (l_prev + b_prev)
        
        # 4. Update Trend: blends observed change (l_t - l_prev) with old trend
        b_t = beta * (l_t - l_prev) + (1 - beta) * b_prev
        
        levels.append(float(l_t))
        
    return levels