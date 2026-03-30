import numpy as np

def sigmoid(x):
    # 
    # Vectorized sigmoid function.
    # 
    
    # Convertimos la entrada a un array de NumPy para manejar diversos tipos de datos
    x = np.asarray(x, dtype=float)
    
    # Aplicamos la fórmula: 1 / (1 + e^-x)
    return 1 / (1 + np.exp(-x))