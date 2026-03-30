import numpy as np

def dropout(x, p=0.5, rng=None):
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """
    x = np.asarray(x, dtype=float)
    
    if rng is None:
        rng = np.random.default_rng()
        
    # Hint 1: Generar valores aleatorios entre 0 y 1
    random_values = rng.random(x.shape)
    
    # Hint 2: Crear el patrón con 0 para dropped y 1/(1-p) para los que se quedan
    # Si random_value < (1-p), se queda.
    scale = 1.0 / (1.0 - p)
    dropout_pattern = np.where(random_values < (1.0 - p), scale, 0.0)
    
    # Aplicar el patrón directamente
    output = x * dropout_pattern
    
    return output, dropout_pattern