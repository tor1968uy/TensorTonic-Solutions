import torch
import torch.nn.functional as F

def sgns_loss(center_vec: torch.Tensor, pos_vec: torch.Tensor, neg_vecs: torch.Tensor) -> torch.Tensor:
    """
    Returns a scalar torch.Tensor: the SGNS loss.
    """
    # 1. Positive pair score: Compute dot product between center and positive target vector
    # Shape: scalar
    pos_score = torch.dot(center_vec, pos_vec)
    
    # 2. Negative pairs scores: Compute dot products between center and all negative vectors at once
    # Shape: (k,)
    neg_scores = torch.mv(neg_vecs, center_vec)
    
    # 3. Calculate stable positive loss term: -log(σ(pos_score)) = softplus(-pos_score)
    pos_loss = F.softplus(-pos_score)
    
    # 4. Calculate stable negative loss term: -log(σ(-neg_score)) = softplus(neg_score)
    # Sum the errors across all k negative samples
    neg_loss = torch.sum(F.softplus(neg_scores))
    
    # Total SGNS loss
    return pos_loss + neg_loss