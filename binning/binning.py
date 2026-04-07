import math

def binning(values, num_bins):
    """
    Assign each value to an equal-width bin index (0 to num_bins - 1).
    Returns a list of integers.
    """
    if not values:
        return []
    
    min_val = min(values)
    max_val = max(values)
    total_range = max_val - min_val
    
    # 1. Handle edge case: all values are the same
    if total_range == 0:
        return [0] * len(values)
    
    # 2. Calculate the width of each bin
    bin_width = total_range / num_bins
    
    bin_indices = []
    for val in values:
        # 3. Calculate the bin index
        # index = floor((val - min_val) / bin_width)
        index = int((val - min_val) // bin_width)
        
        # 4. Handle the boundary case: the maximum value falls into the last bin
        # Without this, max_val would result in index == num_bins
        if index >= num_bins:
            index = num_bins - 1
            
        bin_indices.append(index)
        
    return bin_indices