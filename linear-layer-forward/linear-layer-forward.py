def linear_layer_forward(X, W, b):
    """
    Compute the forward pass of a linear layer: Y = XW + b.
    Returns: A list of lists of floats (n_samples x d_out).
    """
    n_samples = len(X)
    d_in = len(X[0])
    d_out = len(W[0])
    
    # 1. Initialize the output matrix Y with zeros
    # Shape: (n_samples x d_out)
    Y = [[0.0 for _ in range(d_out)] for _ in range(n_samples)]
    
    # 2. Perform matrix multiplication: X @ W
    for i in range(n_samples):
        for j in range(d_out):
            # Compute dot product of X[i] and W[:, j]
            dot_product = 0.0
            for k in range(d_in):
                dot_product += X[i][k] * W[k][j]
            
            # 3. Add the bias and store in the output matrix
            # Note: b[j] is the bias for the j-th output neuron
            Y[i][j] = dot_product + b[j]
            
    return Y