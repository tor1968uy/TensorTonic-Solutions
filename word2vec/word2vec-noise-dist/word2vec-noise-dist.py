import torch

def noise_distribution(counts: torch.Tensor, alpha: float = 0.75) -> torch.Tensor:
    """
    Returns torch.Tensor of shape (vocab_size,), a probability distribution that sums to 1.
    """
    # 1. Cast counts to double precision float64 for stable accumulation
    counts_double = counts.to(dtype=torch.float64)
    
    # 2. Raise each count to the power of alpha
    smoothed_counts = torch.pow(counts_double, alpha)
    
    # 3. Calculate the normalization factor (the partition function)
    partition_sum = torch.sum(smoothed_counts)
    
    # 4. Normalize to make it a valid probability distribution
    distribution = smoothed_counts / partition_sum
    
    return distribution
