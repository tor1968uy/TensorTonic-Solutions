import math

def cosine_embedding_loss(x1, x2, label, margin):
    """
    Compute cosine embedding loss for a pair of vectors.
    Returns: float - the calculated loss value.
    """
    # 1. Compute the dot product: sum(a_i * b_i)
    dot_product = sum(a * b for a, b in zip(x1, x2))
    
    # 2. Compute the L2 norms: sqrt(sum(a_i^2))
    norm_x1 = math.sqrt(sum(a * a for a in x1))
    norm_x2 = math.sqrt(sum(b * b for b in x2))
    
    # 3. Compute Cosine Similarity
    # cos(theta) = (x1 . x2) / (||x1|| * ||x2||)
    cos_sim = dot_product / (norm_x1 * norm_x2)
    
    # 4. Compute loss based on label
    if label == 1:
        # For similar pairs, we want cos_sim to be 1.0 (loss = 0)
        loss = 1 - cos_sim
    else:
        # For dissimilar pairs, we want cos_sim to be less than the margin.
        # If cos_sim is already smaller than the margin, loss is 0.
        loss = max(0.0, cos_sim - margin)
        
    return float(loss)