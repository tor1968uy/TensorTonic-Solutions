def precision_recall_at_k(recommended, relevant, k):
    """
    Compute precision@k and recall@k.
    Returns: [precision, recall] as a list of two floats.
    """
    # 1. Get the top-k recommendations
    top_k = recommended[:k]
    
    # 2. Convert relevant items to a set for O(1) average lookup time
    relevant_set = set(relevant)
    
    # 3. Count the number of 'hits' (recommended items that are relevant)
    hits = 0
    for item in top_k:
        if item in relevant_set:
            hits += 1
            
    # 4. Calculate Precision@K: 
    # What fraction of the Top-K items are actually relevant?
    precision = hits / k
    
    # 5. Calculate Recall@K: 
    # What fraction of all relevant items did we manage to find in the Top-K?
    recall = hits / len(relevant)
    
    return [float(precision), float(recall)]