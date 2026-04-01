def min_max_scaling(data):
    import numpy as np
    data = np.asarray(data, dtype=float)
    
    col_min = data.min(axis=0)
    col_max = data.max(axis=0)
    scale = col_max - col_min
    
    # Evitar división por cero en columnas constantes
    scale[scale == 0] = 1
    
    return ((data - col_min) / scale).tolist()