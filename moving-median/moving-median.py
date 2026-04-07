def moving_median(values, window_size):
    """
    Compute the rolling median for each window position.
    Returns: a list of floats of length n - window_size + 1.
    """
    n = len(values)
    k = window_size
    results = []
    
    # 1. Slide the window from the start of the list
    # The last window starts at index n - k
    for i in range(n - k + 1):
        # 2. Extract and sort the current window slice
        window = sorted(values[i : i + k])
        
        # 3. Calculate the median
        if k % 2 == 1:
            # Odd window: middle element
            median = window[k // 2]
        else:
            # Even window: average of the two middle elements
            m1 = window[k // 2 - 1]
            m2 = window[k // 2]
            median = (m1 + m2) / 2.0
            
        results.append(float(median))
        
    return results