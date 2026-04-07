def item_cf_predict(user_ratings, item_similarities, target):
    """
    Predict a user's rating for a target item based on their other ratings.
    Returns a float (the predicted rating).
    """
    weighted_sum = 0.0
    sum_of_weights = 0.0
    
    # Iterate through all items to find contributors
    for i in range(len(user_ratings)):
        # Skip criteria:
        # 1. Don't compare the target item to itself
        # 2. Skip items the user hasn't rated (rating == 0)
        # 3. Skip items with no positive similarity (si <= 0)
        if i == target or user_ratings[i] == 0 or item_similarities[i] <= 0:
            continue
            
        similarity = item_similarities[i]
        rating = user_ratings[i]
        
        # Accumulate the weighted rating and the weight itself
        weighted_sum += similarity * rating
        sum_of_weights += similarity
        
    # If no items qualified (denominator is 0), return 0.0
    if sum_of_weights == 0:
        return 0.0
        
    # Return the weighted average
    return float(weighted_sum / sum_of_weights)