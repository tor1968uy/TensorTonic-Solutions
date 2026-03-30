import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    m, n = X.shape
    w = np.zeros(n)
    b = 0.0
    
    for _ in range(steps):
        # Usamos tu versión estable de la sigmoide
        linear_model = np.dot(X, w) + b
        y_pred = _sigmoid(linear_model)  # <--- Aquí la magia
        
        # Gradientes
        dw = (1 / m) * np.dot(X.T, (y_pred - y))
        db = (1 / m) * np.sum(y_pred - y)
        
        # Actualización
        w -= lr * dw
        b -= lr * db
        
    return w, b