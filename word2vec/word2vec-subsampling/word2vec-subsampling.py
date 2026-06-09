import torch

def subsample_keep_probs(counts: torch.Tensor, t: float = 1e-5) -> torch.Tensor:
    """
    Returns torch.Tensor of shape (vocab_size,) with the keep-probability for each word.
    """
    # 1. Cast counts to float to avoid integer division
    counts_float = counts.to(dtype=torch.float32)
    
    # 2. Compute total count N and frequencies f(w) = count(w) / N
    total_count = torch.sum(counts_float)
    f = counts_float / total_count
    
    # 3. Calculate the formula: sqrt(t / f(w))
    # Note: If counts can contain 0, we can add a small epsilon to prevent division by zero,
    # but the constraints guarantee a positive sum and valid frequencies.
    keep_probs = torch.sqrt(t / f)
    
    # 4. Clamp the maximum probability to 1.0 (so rare words where f(w) <= t saturate at 1.0)
    return torch.clamp(keep_probs, max=1.0)