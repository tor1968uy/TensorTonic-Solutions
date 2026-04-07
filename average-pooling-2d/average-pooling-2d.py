import numpy as np

def average_pooling_2d(X, pool_size):
    """
    Apply 2D average pooling with non-overlapping windows.
    Returns a 2D list of floats.
    """
    # Convert to numpy array for easier slicing
    X_np = np.asarray(X, dtype=float)
    H, W = X_np.shape
    
    # Calculate output dimensions (discarding remainders)
    H_out = H // pool_size
    W_out = W // pool_size
    
    # Initialize the output matrix
    # Using np.zeros and then converting to list at the end
    output = np.zeros((H_out, W_out))
    
    for i in range(H_out):
        for j in range(W_out):
            # Define the window boundaries
            # Stride is equal to pool_size for non-overlapping windows
            row_start = i * pool_size
            row_end = row_start + pool_size
            col_start = j * pool_size
            col_end = col_start + pool_size
            
            # Extract the window and compute the mean
            window = X_np[row_start:row_end, col_start:col_end]
            output[i, j] = np.mean(window)
            
    # Return as a 2D list of floats as per requirements
    return output.tolist()