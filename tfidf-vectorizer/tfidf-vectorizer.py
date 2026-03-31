import numpy as np
from collections import Counter
import math

def tfidf_vectorizer(documents):
    """
    Build TF-IDF matrix with Normalized Term Frequency (count / doc_length).
    """
    if not documents:
        return np.array([]), []

    # 1. Tokenización
    tokenized_docs = [doc.lower().split() for doc in documents]
    
    # 2. Vocabulario único y ordenado
    all_words = set()
    for doc in tokenized_docs:
        all_words.update(doc)
    vocabulary = sorted(list(all_words))
    
    vocab_size = len(vocabulary)
    num_docs = len(documents)
    
    if vocab_size == 0:
        return np.zeros((num_docs, 0)), []

    word_to_idx = {word: i for i, word in enumerate(vocabulary)}

    # 3. Calcular Document Frequency (DF)
    df_counts = Counter()
    for doc in tokenized_docs:
        unique_words_in_doc = set(doc)
        for word in unique_words_in_doc:
            df_counts[word] += 1

    # 4. Calcular IDF: ln(N / df)
    idf = np.zeros(vocab_size)
    for i, word in enumerate(vocabulary):
        idf[i] = math.log(num_docs / df_counts[word])

    # 5. Construir Matriz TF-IDF con TF Normalizado
    tfidf_matrix = np.zeros((num_docs, vocab_size))
    
    for d_idx, doc in enumerate(tokenized_docs):
        doc_length = len(doc)
        if doc_length == 0:
            continue
            
        term_counts = Counter(doc)
        for word, count in term_counts.items():
            if word in word_to_idx:
                w_idx = word_to_idx[word]
                # TF Normalizado: frecuencia / total de palabras en el doc
                tf = count / doc_length
                tfidf_matrix[d_idx, w_idx] = tf * idf[w_idx]

    return tfidf_matrix, vocabulary

