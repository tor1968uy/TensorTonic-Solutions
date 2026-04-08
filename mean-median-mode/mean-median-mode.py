import numpy as np
from collections import Counter

def mean_median_mode(x):
    """
    Compute mean, median, and mode of a 1D numeric array.
    """
    # Convert input to a numpy array for statistical functions
    data = np.array(x)
    
    # 1. Compute Mean
    mean_val = np.mean(data)
    
    # 2. Compute Median
    median_val = np.median(data)
    
    # 3. Compute Mode
    counts = Counter(x)
    max_freq = max(counts.values())
    
    # Find all candidates with the max frequency
    modes = [val for val, freq in counts.items() if freq == max_freq]
    
    # Per requirements, select the smallest value among the most frequent
    mode_val = min(modes)
    
    # Return as a tuple of floats
    return float(mean_val), float(median_val), float(mode_val)