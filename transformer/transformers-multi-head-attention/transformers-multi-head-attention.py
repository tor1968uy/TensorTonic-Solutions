import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Calcula Multi-Head Attention utilizando proyecciones lineales,
    división en cabezas paralelas y proyección final.
    """
    batch_size, seq_len, d_model = Q.shape
    d_k = d_model // num_heads

    # 1. Proyecciones Lineales (QW, KW, VW)
    # Forma resultante: (batch, seq_len, d_model)
    q_proj = np.dot(Q, W_q)
    k_proj = np.dot(K, W_k)
    v_proj = np.dot(V, W_v)

    # 2. Dividir en cabezas y reordenar dimensiones
    # De (batch, seq, d_model) a (batch, num_heads, seq, d_k)
    def split_heads(x):
        x = x.reshape(batch_size, seq_len, num_heads, d_k)
        return x.transpose(0, 2, 1, 3)

    q_heads = split_heads(q_proj)
    k_heads = split_heads(k_proj)
    v_heads = split_heads(v_proj)

    # 3. Scaled Dot-Product Attention por cada cabeza
    # Puntuaciones: (batch, num_heads, seq, seq)
    scores = np.matmul(q_heads, k_heads.transpose(0, 1, 3, 2)) / np.sqrt(d_k)
    weights = softmax(scores, axis=-1)
    
    # Contexto: (batch, num_heads, seq, d_k)
    attention_output = np.matmul(weights, v_heads)

    # 4. Concatenación de cabezas
    # De (batch, num_heads, seq, d_k) a (batch, seq, d_model)
    concat_output = attention_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)

    # 5. Proyección Final (WO)
    return np.dot(concat_output, W_o)