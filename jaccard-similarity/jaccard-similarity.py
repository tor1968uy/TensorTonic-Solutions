def jaccard_similarity(set_a, set_b):
    """
    Compute the Jaccard similarity between two item sets.
    Returns: float - the Jaccard coefficient between 0.0 and 1.0.
    """
    # 1. Convert lists to sets to remove duplicates and enable set operations
    s1 = set(set_a)
    s2 = set(set_b)
    
    # 2. Compute the intersection (items in both) and union (all unique items)
    intersection = s1.intersection(s2)
    union = s1.union(s2)
    
    # 3. Handle the edge case where both sets are empty
    # This prevents a DivisionByZero error
    if not union:
        return 0.0
        
    # 4. Jaccard Index = size of intersection / size of union
    return float(len(intersection) / len(union))