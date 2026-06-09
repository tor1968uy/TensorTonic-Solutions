import torch
import torch.nn.functional as F

def cbow_forward(context_ids: torch.Tensor, target_id: int, W_in: torch.Tensor, W_out: torch.Tensor) -> torch.Tensor:
    """
    Returns a scalar torch.Tensor: the CBOW cross-entropy loss for predicting target_id from the averaged context.
    """
    # 1. Look up the input embeddings for context words and average them
    # Shape of W_in[context_ids]: (m, D) -> mean(dim=0) -> Shape of h: (D,)
    h = torch.mean(W_in[context_ids], dim=0)
    
    # 2. Score every word in the vocabulary (Compute raw logits)
    # W_out shape: (vocab_size, D), h shape: (D,) -> logits shape: (vocab_size,)
    # We can use matrix-vector multiplication (torch.mv) or matrix multiplication (@)
    logits = torch.mv(W_out, h)
    
    # 3. Apply log-softmax across the full vocabulary to get stable log probabilities
    log_probs = F.log_softmax(logits, dim=0)
    
    # 4. Extract the negative log-probability of our specific target word index
    loss = -log_probs[target_id]
    
    return loss