import torch

def sgns_sgd_step(W_in: torch.Tensor, W_out: torch.Tensor, center_id: int, pos_id: int,
                  neg_ids: torch.Tensor, lr: float) -> tuple:
    """
    Returns tuple (W_in_updated, W_out_updated), each the same shape as the inputs, after one SGNS SGD step.
    """
    # Clone the input matrices to avoid modifying the originals in-place
    W_in_updated = W_in.clone()
    W_out_updated = W_out.clone()
    
    # 1. Snapshot the pre-update center vector from W_in
    v_c = W_in[center_id].clone()
    
    # 2. Snapshot the pre-update context vectors from W_out
    u_o = W_out[pos_id].clone()
    
    # 3. Process the positive target word
    score_o = torch.dot(v_c, u_o)
    sigma_o = torch.sigmoid(score_o)
    coeff_o = sigma_o - 1.0
    
    # Compute the gradient contributions for the center vector
    grad_v_c = coeff_o * u_o
    
    # Update the positive context word vector in W_out
    grad_u_o = coeff_o * v_c
    W_out_updated[pos_id] -= lr * grad_u_o
    
    # 4. Process negative noise words
    # We loop through negative IDs individually to properly accumulate duplicates 
    for neg_id in neg_ids:
        neg_id_item = neg_id.item()
        u_n = W_out[neg_id_item].clone()
        
        score_n = torch.dot(v_c, u_n)
        sigma_n = torch.sigmoid(score_n)
        
        # Accumulate the gradient contribution for the center vector
        grad_v_c += sigma_n * u_n
        
        # Update the negative context word vector in W_out (accumulates if ID repeats)
        grad_u_n = sigma_n * v_c
        W_out_updated[neg_id_item] -= lr * grad_u_n
        
    # 5. Apply the accumulated gradient update to the center vector in W_in
    W_in_updated[center_id] -= lr * grad_v_c
    
    return W_in_updated, W_out_updated