def differencing(series, order):
    """
    Apply d-th order differencing to the time series.
    Returns: a list of numbers of length n - order.
    """
    # 1. Start with the original series as our working data
    current_series = list(series)
    
    # 2. Perform the differencing operation 'order' times
    for _ in range(order):
        # 3. Create a new list for this round of differences
        # Each element is (current_element - previous_element)
        diff_round = []
        for i in range(1, len(current_series)):
            delta = current_series[i] - current_series[i-1]
            diff_round.append(delta)
        
        # 4. Update working data for the next round (if any)
        current_series = diff_round
        
    return current_series