import numpy as np

def random_forest_vote(predictions):
    # Convertimos a array de NumPy
    # La estructura es (n_trees, n_samples)
    preds_array = np.array(predictions)
    
    # Trasponemos para tener (n_samples, n_trees) 
    # Así podemos iterar sobre las predicciones de cada muestra
    samples_preds = preds_array.T
    
    final_predictions = []
    
    for sample in samples_preds:
        # Contamos la frecuencia de cada etiqueta (clase)
        # np.bincount devuelve un array donde el índice es la clase 
        # y el valor es el número de votos.
        counts = np.bincount(sample.astype(int))
        
        # np.argmax devuelve el ÍNDICE del valor máximo.
        # CRITICAL: En caso de múltiples valores máximos (empate), 
        # argmax SIEMPRE devuelve el primer índice encontrado (el menor).
        # Esto cumple automáticamente con el requisito de "break ties by smallest label".
        winning_class = np.argmax(counts)
        
        final_predictions.append(int(winning_class))
        
    return final_predictions