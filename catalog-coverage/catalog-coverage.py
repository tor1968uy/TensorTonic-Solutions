def catalog_coverage(recommendations, n_items):
    """
    Compute the catalog coverage of a recommender system.
    Returns: float - fraction of catalog items recommended.
    """
    # 1. Handle edge case: empty catalog
    if n_items == 0:
        return 0.0
    
    # 2. Use a set to collect all unique recommended items
    # Sets automatically handle deduplication across different users
    unique_recommended = set()
    
    for user_list in recommendations:
        for item_id in user_list:
            unique_recommended.add(item_id)
            
    # 3. Calculate coverage: (Unique Items Recommended) / (Total Catalog Size)
    coverage = len(unique_recommended) / n_items
    
    return float(coverage)