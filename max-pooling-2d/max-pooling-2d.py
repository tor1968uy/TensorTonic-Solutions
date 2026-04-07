def max_pooling_2d(X, pool_size):
    """
    Apply 2D max pooling with non-overlapping windows.
    X: 2D list of numbers
    pool_size: integer representing the height and width of the pooling window
    """
    H = len(X)
    W = len(X[0])
    
    # 1. Calculate output dimensions (integer division)
    # H_out = floor(H / pool_size)
    h_out = H // pool_size
    w_out = W // pool_size
    
    pooled_output = []
    
    # 2. Iterate over the output grid
    for i in range(h_out):
        new_row = []
        for j in range(w_out):
            # 3. Define the window boundaries in the input matrix X
            # Starting coordinates: (i * p, j * p)
            start_row = i * pool_size
            start_col = j * pool_size
            
            # Initialize max_val with the first element of the window 
            # or a very small number
            current_max = X[start_row][start_col]
            
            # 4. Find the maximum value within the p x p pool
            for r in range(start_row, start_row + pool_size):
                for c in range(start_col, start_col + pool_size):
                    if X[r][c] > current_max:
                        current_max = X[r][c]
            
            new_row.append(current_max)
        pooled_output.append(new_row)
            
    return pooled_output