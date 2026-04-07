import math

def novelty_score(recommendations, item_counts, n_users):
    """
    Compute the average novelty of a recommendation list.
    Returns a float representing the average bits of self-information.
    """
    if not recommendations:
        return 0.0
    
    total_novelty = 0.0
    num_items = len(recommendations)
    
    for item_idx in recommendations:
        # 1. Get the interaction count for the recommended item
        count_i = item_counts[item_idx]
        
        # 2. Compute popularity (probability of interaction)
        # p(i) = count_i / n_users
        popularity = count_i / n_users
        
        # 3. Compute self-information: -log2(p(i))
        # This is measured in bits. 
        # Rare items (low p) result in high self-information.
        item_novelty = -math.log2(popularity)
        
        total_novelty += item_novelty
        
    # 4. Return the average novelty across the recommendation list
    return float(total_novelty / num_items)