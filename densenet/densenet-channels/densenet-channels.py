import math
import torch

def densenet_channel_counts(
    stem_channels: int, 
    growth_rate: int, 
    block_layers: list[int], 
    compression: float
) -> torch.Tensor:
    """
    Returns a 1D int64 torch.Tensor of channel counts at each stage.
    """
    current_channels = stem_channels
    history = [current_channels]
    num_blocks = len(block_layers)
    
    for i, n in enumerate(block_layers):
        # 1. Block Growth: Each layer adds `growth_rate` channels via concatenation
        current_channels += n * growth_rate
        history.append(current_channels)
        
        # 2. Transition Layer: Compresses channels, omitted after the final block
        if i < num_blocks - 1:
            current_channels = math.floor(compression * current_channels)
            history.append(current_channels)
            
    return torch.tensor(history, dtype=torch.int64)