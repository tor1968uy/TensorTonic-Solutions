import numpy as np

def softmax(x, axis=-1):
    """Provided: Softmax function."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return gamma * (x - mean) / np.sqrt(var + eps) + beta

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    batch_size, seq_len, d_model = Q.shape
    d_k = d_model // num_heads
    
    # Linear Projections
    q = np.dot(Q, W_q).reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
    k = np.dot(K, W_k).reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
    v = np.dot(V, W_v).reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
    
    # Scaled Dot-Product Attention
    scores = np.matmul(q, k.transpose(0, 1, 3, 2)) / np.sqrt(d_k)
    weights = softmax(scores)
    context = np.matmul(weights, v) # (batch, num_heads, seq, d_k)
    
    # Concatenate and Project
    concat = context.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)
    return np.dot(concat, W_o)

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    hidden = np.maximum(0, np.dot(x, W1) + b1)
    return np.dot(hidden, W2) + b2

def encoder_block(x: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                  W_o: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray,
                  b2: np.ndarray, gamma1: np.ndarray, beta1: np.ndarray,
                  gamma2: np.ndarray, beta2: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Complete encoder block: MHA + Residual + LN, then FFN + Residual + LN.
    """
    # 1. Multi-Head Self-Attention Sublayer
    # Self-attention means Q, K, and V all come from the same input x
    attn_out = multi_head_attention(x, x, x, W_q, W_k, W_v, W_o, num_heads)
    
    # Residual + LayerNorm
    x_prime = layer_norm(x + attn_out, gamma1, beta1)
    
    # 2. Feed-Forward Sublayer
    ff_out = feed_forward(x_prime, W1, b1, W2, b2)
    
    # Residual + LayerNorm
    output = layer_norm(x_prime + ff_out, gamma2, beta2)
    
    return output