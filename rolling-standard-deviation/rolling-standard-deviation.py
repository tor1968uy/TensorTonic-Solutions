import math

def rolling_std(values, window_size):
    """
    Compute the rolling population standard deviation.
    Returns a list of floats of length len(values) - window_size + 1.
    """
    n = len(values)
    output_length = n - window_size + 1
    results = []
    
    # Slide the window across the values
    for i in range(output_length):
        # 1. Extract the current window
        window = values[i : i + window_size]
        
        # 2. Calculate the mean (mu) of the window
        window_mean = sum(window) / window_size
        
        # 3. Calculate the population variance
        # Sum of squared differences from the mean, divided by N
        squared_diff_sum = sum((x - window_mean) ** 2 for x in window)
        variance = squared_diff_sum / window_size
        
        # 4. Standard deviation is the square root of variance
        std_dev = math.sqrt(variance)
        results.append(float(std_dev))
        
    return results