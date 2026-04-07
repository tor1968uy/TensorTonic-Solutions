import numpy as np

def adjusted_cosine_similarity(ratings_matrix, item_i, item_j):
    """
    Compute adjusted cosine similarity between two items in a ratings matrix.
    """
    matrix = np.asarray(ratings_matrix, dtype=float)
    num_users = matrix.shape[0]
    
    # 1. Calculate the mean rating for each user (using only non-zero ratings)
    user_means = np.zeros(num_users)
    for u in range(num_users):
        user_ratings = matrix[u, :]
        rated_indices = user_ratings != 0
        if np.any(rated_indices):
            user_means[u] = np.mean(user_ratings[rated_indices])
    
    # 2. Identify users who rated both item_i AND item_j
    # Only these users contribute to the similarity calculation
    both_rated_mask = (matrix[:, item_i] != 0) & (matrix[:, item_j] != 0)
    
    if not np.any(both_rated_mask):
        return 0.0
    
    # 3. Extract ratings and subtract user means (centering)
    # r_ui - mean_u and r_uj - mean_u
    diff_i = matrix[both_rated_mask, item_i] - user_means[both_rated_mask]
    diff_j = matrix[both_rated_mask, item_j] - user_means[both_rated_mask]
    
    # 4. Compute numerator: sum of (diff_i * diff_j)
    numerator = np.sum(diff_i * diff_j)
    
    # 5. Compute denominator: sqrt(sum(diff_i^2)) * sqrt(sum(diff_j^2))
    denom_i = np.sum(diff_i**2)
    denom_j = np.sum(diff_j**2)
    denominator = np.sqrt(denom_i) * np.sqrt(denom_j)
    
    # Handle zero denominator to avoid division by zero
    if denominator == 0:
        return 0.0
    
    return float(numerator / denominator)