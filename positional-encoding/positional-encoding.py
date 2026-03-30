import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    """
    # 1. Creamos una matriz de ceros de la forma deseada
    pe = np.zeros((seq_len, d_model))
    
    # 2. Generamos el vector de posiciones (0, 1, ..., seq_len-1)
    # Lo redimensionamos a (seq_len, 1) para poder multiplicar por filas
    position = np.arange(seq_len)[:, np.newaxis]
    
    # 3. Calculamos el término de la frecuencia (el denominador)
    # Solo necesitamos calcularlo para cada par de dimensiones (i)
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(base) / d_model))
    
    # 4. Aplicamos seno a los índices pares (0, 2, 4...)
    pe[:, 0::2] = np.sin(position * div_term)
    
    # 5. Aplicamos coseno a los índices impares (1, 3, 5...)
    # Si d_model es impar, hay que asegurarse de que los tamaños coincidan
    pe[:, 1::2] = np.cos(position * div_term[:d_model//2])
    
    return pe