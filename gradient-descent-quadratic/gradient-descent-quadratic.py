import numpy as np

def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    """
    Return final x after 'steps' iterations.
    """
    x = x0
    
    for _ in range(steps):
        # 1. Calcular el gradiente en el punto actual
        # Derivada de ax^2 + bx + c es 2ax + b
        gradient = 2 * a * x + b
        
        # 2. Actualizar x moviéndonos en dirección opuesta al gradiente
        x = x - lr * gradient
        
    return x