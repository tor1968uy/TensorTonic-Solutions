import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Genera una matriz de codificación posicional sinusoidal.
    """
    # 1. Inicializar la matriz con ceros: forma (largo de secuencia, dimensión del modelo)
    pe = np.zeros((seq_length, d_model))
    
    # 2. Crear un vector de posiciones (0, 1, 2, ..., seq_length-1)
    # Lo transformamos en columna (seq_length, 1) para facilitar el broadcasting
    position = np.arange(seq_length).reshape(-1, 1)
    
    # 3. Calcular el término divisor (las frecuencias)
    # Usamos el truco del espacio logarítmico para mayor estabilidad numérica
    # i va de 0 a d_model en pasos de 2 (solo índices pares)
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    
    # 4. Aplicar seno a los índices pares (0, 2, 4...)
    pe[:, 0::2] = np.sin(position * div_term)
    
    # 5. Aplicar coseno a los índices impares (1, 3, 5...)
    pe[:, 1::2] = np.cos(position * div_term)
    
    return pe