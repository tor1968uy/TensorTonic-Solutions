def winsorize(values, lower_pct, upper_pct):
    import numpy as np
    v = np.asarray(values, dtype=float)
    
    lower_bound = np.percentile(v, lower_pct)
    upper_bound = np.percentile(v, upper_pct)
    
    return np.clip(v, lower_bound, upper_bound).tolist()