def seasonal_average(series, period):
    """
    Compute the average value for each position in the seasonal cycle.
    Returns a list of floats of length equal to period.
    """
    seasonal_averages = []
    
    # Iterate through each position in the seasonal cycle
    for p in range(period):
        # Gather all values at this seasonal position: p, p+period, p+2*period, ...
        # The range(p, len(series), period) handles the 'step' for us
        position_values = [series[i] for i in range(p, len(series), period)]
        
        # Calculate the mean for this position
        if position_values:
            avg = sum(position_values) / len(position_values)
            seasonal_averages.append(float(avg))
        else:
            # Should not be reached based on constraints
            seasonal_averages.append(0.0)
            
    return seasonal_averages