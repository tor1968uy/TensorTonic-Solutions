import numpy as np

def softmax(x):
    # CONVERSIÓN CRUCIAL: Asegura que x sea un arreglo de NumPy
    x = np.array(x)
    
    # Ahora .ndim funcionará sin errores
    if x.ndim == 1:
        x_max = np.max(x)
    else:
        x_max = np.max(x, axis=-1, keepdims=True)
    
    exps = np.exp(x - x_max)
    
    if x.ndim == 1:
        sum_exps = np.sum(exps)
    else:
        sum_exps = np.sum(exps, axis=-1, keepdims=True)
        
    return exps / sum_exps

# Prueba con una lista (ahora funcionará)
x = [1, 2, 3]
print(softmax(x))