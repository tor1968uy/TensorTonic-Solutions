def cumulative_returns(returns):
    """
    Compute the cumulative return at each time step.
    Returns: a list of floats representing the total return since inception.
    """
    cumulative_results = []
    # 1. Start with a base wealth factor of 1.0 (representing 100% of capital)
    current_wealth = 1.0
    
    for r in returns:
        # 2. Update wealth factor multiplicatively
        # If r is 0.05 (5%), we multiply by 1.05
        # If r is -0.10 (-10%), we multiply by 0.90
        current_wealth *= (1 + r)
        
        # 3. Cumulative return is the total gain/loss relative to the start
        # Wealth of 1.155 means a cumulative return of 0.155 (15.5%)
        cumulative_results.append(current_wealth - 1.0)
        
    return cumulative_results