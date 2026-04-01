def autocorrelation(series, max_lag):
    import numpy as np
    x = np.array(series, dtype=float)
    n = len(x)
    mean = x.mean()
    x_centered = x - mean
    var = np.sum(x_centered ** 2) / n

    if var == 0:
        return [1.0] + [0.0] * max_lag

    result = []
    for lag in range(max_lag + 1):
        cov = np.sum(x_centered[:n - lag] * x_centered[lag:]) / n
        result.append(cov / var)

    return result