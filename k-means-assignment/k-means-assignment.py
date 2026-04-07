def k_means_assignment(points, centroids):
    """
    Assign each data point to the nearest centroid.
    Returns: a list of integer cluster indices.
    """
    assignments = []
    
    for p in points:
        best_dist = float('inf')
        best_idx = 0
        
        for idx, c in enumerate(centroids):
            # 1. Compute squared Euclidean distance: sum((p_d - c_d)^2)
            current_sq_dist = 0
            for d in range(len(p)):
                current_sq_dist += (p[d] - c[d]) ** 2
            
            # 2. Update best assignment if this centroid is closer
            # Using strict '<' ensures tie-breaking by the smallest index
            if current_sq_dist < best_dist:
                best_dist = current_sq_dist
                best_idx = idx
        
        assignments.append(best_idx)
            
    return assignments