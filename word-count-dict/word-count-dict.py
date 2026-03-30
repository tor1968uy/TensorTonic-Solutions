def word_count_dict(sentences):
    """
    Returns: dict[str, int] - global word frequency across all sentences
    """
    # Hint 1: Initialize an empty dictionary
    word_counts = {}
    
    # Recorremos la lista principal (que contiene sublistas)
    for sentence in sentences:
        
        # SI SENTENCE ES UNA LISTA: Iteramos directamente sobre sus elementos
        # (Aquí es donde fallaba el .split())
        for token in sentence:
            if token in word_counts:
                word_counts[token] += 1
            else:
                word_counts[token] = 1
                
    return word_counts