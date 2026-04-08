import math

def log_transform(values):
    """
    Apply the log1p transformation to each value in the list.
    Returns: A list of floats.
    """
    # 1. Use a list comprehension to apply the transform to each value
    # math.log1p(x) is mathematically equivalent to ln(1 + x)
    transformed_values = [float(math.log1p(x)) for x in values]
    
    return transformed_values