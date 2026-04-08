def k_means_centroid_update(points, assignments, k):
    """
    Compute new centroids as the mean of assigned points.
    Returns: A list of k lists, where each inner list is a centroid.
    """
    if not points:
        return []
        
    dim = len(points[0])
    # 1. Initialize sums for each dimension of each centroid and a counter
    # cluster_sums: [[dim1_sum, dim2_sum, ...], [dim1_sum, dim2_sum, ...], ...]
    cluster_sums = [[0.0] * dim for _ in range(k)]
    counts = [0] * k
    
    # 2. Accumulate the coordinates and counts for each cluster
    for point, cluster_id in zip(points, assignments):
        counts[cluster_id] += 1
        for d in range(dim):
            cluster_sums[cluster_id][d] += point[d]
            
    # 3. Calculate the mean for each cluster
    new_centroids = []
    for i in range(k):
        if counts[i] == 0:
            # Handle empty clusters as per requirements
            new_centroids.append([0.0] * dim)
        else:
            # Divide the sum of each dimension by the number of points in the cluster
            mean_vector = [dimension_sum / counts[i] for dimension_sum in cluster_sums[i]]
            new_centroids.append(mean_vector)
            
    return new_centroids