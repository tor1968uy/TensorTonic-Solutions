import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Calcula el Scaled Dot-Product Attention.
    """
    # 1. Obtener la dimensión de las claves (d_k)
    # K tiene forma (batch, seq_len_k, d_k)
    d_k = K.shape[-1]
    
    # 2. Calcular los puntajes de atención (Q * K^T)
    # Transponemos las últimas dos dimensiones de K para la multiplicación de matrices
    # scores shape: (batch, seq_len_q, seq_len_k)
    scores = torch.matmul(Q, K.transpose(-2, -1))
    
    # 3. Escalar los puntajes por la raíz cuadrada de d_k
    # Esto evita que los productos punto crezcan demasiado y saturen el softmax
    scaled_scores = scores / math.sqrt(d_k)
    
    # 4. Aplicar Softmax a lo largo de la última dimensión (la de las claves)
    # Esto nos da los pesos de atención que suman 1.0
    attention_weights = F.softmax(scaled_scores, dim=-1)
    
    # 5. Calcular la salida final como el producto de pesos y Valores (V)
    # output shape: (batch, seq_len_q, d_v)
    output = torch.matmul(attention_weights, V)
    
    return output