def user_based_cf_prediction(similarities, ratings):
    """
    Predict a rating using user-based collaborative filtering.
    Returns the weighted average of ratings from users with positive similarity.
    """
    weighted_sum = 0.0
    sum_of_weights = 0.0
    
    # 1. Iterate through neighbors (similarity scores and their ratings)
    for sim, rating in zip(similarities, ratings):
        # 2. Filter: Only consider users with positive similarity
        if sim > 0:
            weighted_sum += sim * rating
            sum_of_weights += sim
            
    # 3. Handle edge case: No users with positive similarity
    if sum_of_weights == 0:
        return 0.0
        
    # 4. Compute and return the weighted average
    return float(weighted_sum / sum_of_weights)