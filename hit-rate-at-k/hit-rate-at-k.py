def hit_rate_at_k(recommendations, ground_truth, k):
    """
    Compute the hit rate at K for a set of users.
    
    Args:
        recommendations: List of lists (top-N items predicted for each user)
        ground_truth: List of lists (actual items the user interacted with)
        k: The cutoff for the recommendation list
        
    Returns:
        float: The hit rate (between 0.0 and 1.0)
    """
    if not recommendations or not ground_truth:
        return 0.0
        
    hits = 0
    num_users = len(recommendations)
    
    for i in range(num_users):
        # 1. Consider only the top-K recommended items
        top_k_recs = set(recommendations[i][:k])
        
        # 2. Check if any ground truth item is present in the top-K set
        user_ground_truth = set(ground_truth[i])
        
        # If the intersection is not empty, we have a "Hit"
        if top_k_recs.intersection(user_ground_truth):
            hits += 1
            
    # 3. Hit Rate = Total Hits / Total Users
    return hits / num_users