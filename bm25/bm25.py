import numpy as np
from collections import Counter
import math

def bm25_score(query_tokens, docs, k1=1.2, b=0.75):
    """
    Calcula los puntajes BM25 para un conjunto de documentos y una consulta.
    """
    # 1. Manejo de caso borde: corpus vacío
    if not docs:
        return np.array([], dtype=float)

    N = len(docs)
    # 2. Calcular longitudes de documentos y el promedio (avgdl)
    doc_lengths = np.array([len(d) for d in docs], dtype=float)
    avgdl = np.mean(doc_lengths)
    
    # 3. Pre-calcular frecuencias de términos por documento (TF)
    # Solo procesamos los términos que están en la consulta para mayor eficiencia
    unique_query_terms = list(dict.fromkeys(query_tokens))
    
    # Construimos contadores para cada documento
    doc_counters = [Counter(d) for d in docs]
    
    scores = np.zeros(N, dtype=float)
    
    # Pre-calculamos la parte del denominador que solo depende de la longitud del doc
    # L_norm = (1 - b + b * (|D| / avgdl))
    # Evitamos división por cero si avgdl es 0 (docs vacíos)
    if avgdl > 0:
        len_norm_part = k1 * (1 - b + b * (doc_lengths / avgdl))
    else:
        len_norm_part = np.zeros(N)

    # 4. Iterar sobre términos únicos de la consulta
    for q in unique_query_terms:
        # Frecuencia del término q en cada documento (Vector de tamaño N)
        freqs = np.array([c[q] for c in doc_counters], dtype=float)
        
        # n(q): cantidad de documentos que contienen el término
        nq = np.sum(freqs > 0)
        
        # Si el término no existe en el corpus, su aporte es 0
        if nq == 0:
            continue
            
        # 5. Cálculo de IDF (variante Robertson-Spärck Jones)
        # log((N - nq + 0.5) / (nq + 0.5) + 1)
        idf = math.log((N - nq + 0.5) / (nq + 0.5) + 1)
        
        # 6. Cálculo del componente BM25 para este término (Vectorizado)
        # Numerador: f * (k1 + 1)
        # Denominador: f + k1 * (1 - b + b * |D| / avgdl)
        numerator = freqs * (k1 + 1)
        denominator = freqs + len_norm_part
        
        # Sumamos al score total de cada documento
        scores += idf * (numerator / denominator)

    return scores

# --- Pruebas con los ejemplos del enunciado ---
if __name__ == "__main__":
    # Test 1
    q1 = ["machine", "learning"]
    d1 = [["introduction", "to", "machine", "learning"], ["deep", "learning", "basics"], ["cooking", "pasta", "guide"]]
    print(f"Test 1: {np.round(bm25_score(q1, d1), 5)}")

    # Test 2
    q2 = ["data"]
    d2 = [["data", "science"], ["big", "data", "analytics"], ["cooking", "recipes"]]
    print(f"Test 2: {np.round(bm25_score(q2, d2), 5)}")