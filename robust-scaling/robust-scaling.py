def robust_scaling(values):
    import numpy as np
    v = np.asarray(values, dtype=float)
    n = len(v)
    
    median = np.median(v)
    
    if n < 2:
        return [0.0] * n
    
    sorted_v = np.sort(v)
    lower = sorted_v[:n // 2]
    upper = sorted_v[n - n // 2:]
    q1 = np.median(lower)
    q3 = np.median(upper)
    iqr = q3 - q1
    
    if iqr == 0:
        return [0.0] * n
    
    return ((v - median) / iqr).tolist()