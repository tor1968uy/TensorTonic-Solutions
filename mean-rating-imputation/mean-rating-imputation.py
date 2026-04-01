def mean_rating_imputation(ratings_matrix, mode):
    import numpy as np
    R = np.asarray(ratings_matrix, dtype=float)
    result = R.copy()
    mask = R == 0  # zeros = missing
    
    if mode == 'user':
        # Media por fila (usuario)
        for i in range(R.shape[0]):
            row = R[i]
            known = row[row != 0]
            mean = known.mean() if len(known) > 0 else 0.0
            result[i, mask[i]] = mean
    elif mode == 'item':
        # Media por columna (item)
        for j in range(R.shape[1]):
            col = R[:, j]
            known = col[col != 0]
            mean = known.mean() if len(known) > 0 else 0.0
            result[mask[:, j], j] = mean
    
    return result.tolist()