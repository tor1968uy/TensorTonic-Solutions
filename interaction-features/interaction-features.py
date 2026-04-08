def interaction_features(X):
    """
    Generate pairwise interaction features and append them to the original features.
    Returns: List of lists with d + (d*(d-1)/2) features per sample.
    """
    result = []
    
    for row in X:
        # 1. Start with a copy of the original features
        new_row = list(row)
        num_features = len(row)
        
        # 2. Compute pairwise products x_i * x_j for i < j
        # The outer loop goes through each feature
        for i in range(num_features):
            # The inner loop starts from the next feature to avoid 
            # self-interactions and duplicate pairs (e.g., 1*2 and 2*1)
            for j in range(i + 1, num_features):
                product = row[i] * row[j]
                new_row.append(product)
                
        result.append(new_row)
        
    return result