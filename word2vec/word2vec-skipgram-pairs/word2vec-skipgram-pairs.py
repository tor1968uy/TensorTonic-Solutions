import torch

def skipgram_pairs(token_ids: torch.Tensor, window: int) -> torch.Tensor:
    """
    Returns int64 torch.Tensor of shape (num_pairs, 2).
    """
    n = token_ids.size(0)
    pairs = []
    
    # Iterate through each token as the center word
    for i in range(n):
        # Determine sliding window bounds while preventing out-of-bounds indices
        start = max(0, i - window)
        end = min(n - 1, i + window)
        
        # Gather all valid context tokens inside the window bounds
        for j in range(start, end + 1):
            if i != j:  # Exclude the center word itself
                pairs.append([token_ids[i].item(), token_ids[j].item()])
                
    # If no pairs were generated (e.g., sequence length <= 1), return an empty tensor
    if len(pairs) == 0:
        return torch.zeros((0, 2), dtype=torch.int64)
        
    return torch.tensor(pairs, dtype=torch.int64)