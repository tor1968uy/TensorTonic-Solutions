def lag_features(series, lags):
    """
    Create a lag feature matrix from the time series.
    Returns: A list of lists representing the feature matrix.
    """
    # 1. Determine the maximum lag to know where we can safely start
    max_lag = max(lags)
    n = len(series)
    
    feature_matrix = []
    
    # 2. Iterate from the first valid time step to the end
    # We start at max_lag because any index smaller would have missing lags
    for t in range(max_lag, n):
        row = []
        # 3. For each lag specified, look back from the current time t
        for lag in lags:
            # Value at time t-1, t-2, etc.
            val = series[t - lag]
            row.append(val)
            
        feature_matrix.append(row)
        
    return feature_matrix