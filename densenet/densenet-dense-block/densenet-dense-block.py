import torch
import torch.nn.functional as F

def dense_block(x, layers, growth_rate, eps=1e-5):
    """
    Returns torch.Tensor of shape (N, C + L*growth_rate, H, W).
    Tracks the accumulation of preceding feature maps via concatenation.
    All calculations are evaluated using torch.float64 precision.
    """
    # 1. Force the initial input tensor to high-precision float64
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float64)
    else:
        x = x.to(torch.float64)
        
    # Maintain a list of all active feature maps (initially just the block input x)
    features = [x]
    
    # Helper to parse and reshape 1D BN list/tensor parameters to (1, C, 1, 1)
    def to_4d_tensor(param, device):
        if not isinstance(param, torch.Tensor):
            param = torch.tensor(param, dtype=torch.float64, device=device)
        else:
            param = param.to(dtype=torch.float64, device=device)
        return param.view(1, -1, 1, 1)

    # 2. Iterate sequentially through each layer in the Dense Block
    for layer_dict in layers:
        # Concatenate all preceding feature maps to form the collective input
        current_input = torch.cat(features, dim=1)
        
        # Pull out and parse the current layer parameters
        bn_gamma = layer_dict['bn_gamma']
        bn_beta = layer_dict['bn_beta']
        bn_mean = layer_dict['bn_mean']
        bn_var = layer_dict['bn_var']
        conv_weight = layer_dict['conv_weight']
        
        if not isinstance(conv_weight, torch.Tensor):
            conv_weight = torch.tensor(conv_weight, dtype=torch.float64, device=x.device)
        else:
            conv_weight = conv_weight.to(dtype=torch.float64, device=x.device)
            
        gamma = to_4d_tensor(bn_gamma, x.device)
        beta = to_4d_tensor(bn_beta, x.device)
        mean = to_4d_tensor(bn_mean, x.device)
        var = to_4d_tensor(bn_var, x.device)
        
        # 3. Composite Operation: BN -> ReLU -> 3x3 Conv
        x_norm = (current_input - mean) / torch.sqrt(var + eps)
        x_bn = gamma * x_norm + beta
        x_relu = F.relu(x_bn)
        
        # 3x3 Convolution (padding=1 maintains H and W)
        layer_output = F.conv2d(x_relu, conv_weight, bias=None, stride=1, padding=1)
        
        # Append the new output to the running tracking list
        features.append(layer_output)
        
    # 4. Return the complete concatenated block feature representation
    return torch.cat(features, dim=1)
