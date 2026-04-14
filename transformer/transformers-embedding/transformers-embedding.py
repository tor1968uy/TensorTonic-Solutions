import torch
import torch.nn as nn
import math

def create_embedding_layer(vocab_size: int, d_model: int) -> nn.Embedding:
    """
    Crea una capa de embedding de PyTorch con el tamaño de vocabulario
    y la dimensión de modelo especificados.
    """
    # nn.Embedding actúa como una tabla de búsqueda (lookup table)
    # que almacena vectores de tamaño d_model para cada palabra en el vocabulario.
    return nn.Embedding(vocab_size, d_model)

def embed_tokens(embedding: nn.Embedding, tokens: torch.Tensor, d_model: int) -> torch.Tensor:
    """
    Convierte los índices de tokens en embeddings y aplica el escalado
    especificado en el paper "Attention Is All You Need".
    """
    # 1. Realizar el lookup: convierte cada índice en su vector correspondiente
    # Resultado: (sequence_length, d_model)
    embedded = embedding(tokens)
    
    # 2. Aplicar el escalado por la raíz cuadrada de d_model
    # Esto ayuda a que el entrenamiento sea más estable al normalizar la varianza
    scaled_embeddings = embedded * math.sqrt(d_model)
    
    return scaled_embeddings