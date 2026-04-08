import math

def cyclic_encoding(values, period):
    """
    Encode cyclic features (e.g., hours, days) as sin/cos pairs.
    Returns: A list of [sin, cos] pairs.
    """
    encoded_values = []
    
    for v in values:
        # 1. Calculate the angle in radians
        # Angle = (2 * PI * value) / Period
        # This maps the value to a position on a 360-degree circle
        angle = 2 * math.pi * v / period
        
        # 2. Compute the coordinates on the unit circle
        s = math.sin(angle)
        c = math.cos(angle)
        
        # 3. Append as a [sine, cosine] pair
        encoded_values.append([s, c])
        
    return encoded_values