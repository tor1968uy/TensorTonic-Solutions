def matrix_factorization_sgd_step(U, V, r, lr, reg):
    """
    Performs one SGD update for user and item latent vectors.
    
    Args:
        U (list[float]): User latent vector
        V (list[float]): Item latent vector
        r (float): Actual rating
        lr (float): Learning rate
        reg (float): L2 regularization strength
        
    Returns:
        tuple: (updated_U, updated_V)
    """
    # 1. Compute the dot product (prediction)
    prediction = sum(u * v for u, v in zip(U, V))
    
    # 2. Compute the prediction error
    error = r - prediction
    
    # 3. Compute new U values using original V
    # Formula: U_i_new = U_i + lr * (error * V_i - reg * U_i)
    U_new = [
        u + lr * (error * v - reg * u) 
        for u, v in zip(U, V)
    ]
    
    # 4. Compute new V values using original U
    # Formula: V_i_new = V_i + lr * (error * U_i - reg * V_i)
    V_new = [
        v + lr * (error * u - reg * v) 
        for u, v in zip(U, V)
    ]
    
    return U_new, V_new