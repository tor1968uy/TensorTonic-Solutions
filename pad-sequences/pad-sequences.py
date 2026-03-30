import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L)
    """
    # 1. Determinar la longitud L
    if max_len is None:
        L = max(len(seq) for seq in seqs) if seqs else 0
    else:
        L = max_len
    
    N = len(seqs)
    
    # 2. Crear una matriz llena del valor de relleno (pad_value)
    # Usamos el tipo de dato de la primera secuencia para mantener consistencia
    dtype = np.array(seqs[0]).dtype if seqs and len(seqs[0]) > 0 else int
    padded_matrix = np.full((N, L), pad_value, dtype=dtype)
    
    # 3. Llenar la matriz con las secuencias originales
    for i, seq in enumerate(seqs):
        if len(seq) == 0:
            continue
        
        # Truncamos si la secuencia es más larga que max_len
        actual_len = min(len(seq), L)
        padded_matrix[i, :actual_len] = seq[:actual_len]
        
    return padded_matrix