import math
import torch
import torch.nn.functional as F

def composite_layer(x, bn_gamma, bn_beta, bn_mean, bn_var, conv_weight, eps=1e-5):
    """
    Returns torch.Tensor of shape (N, growth_rate, H, W): BN, ReLU, then a 3x3 same-padding convolution.
    Calculated strictly in float64 precision to prevent tiny output mismatches.
    """
    # 1. Cast primary inputs to high-precision float64 tensors
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float64)
    else:
        x = x.to(torch.float64)
        
    if not isinstance(conv_weight, torch.Tensor):
        conv_weight = torch.tensor(conv_weight, dtype=torch.float64)
    else:
        conv_weight = conv_weight.to(torch.float64)

    # 2. Helper to cleanly convert and reshape BN parameters to (1, C, 1, 1) in float64
    def to_4d_tensor(param):
        if not isinstance(param, torch.Tensor):
            param = torch.tensor(param, dtype=torch.float64, device=x.device)
        else:
            param = param.to(dtype=torch.float64, device=x.device)
        return param.view(1, -1, 1, 1)

    gamma = to_4d_tensor(bn_gamma)
    beta = to_4d_tensor(bn_beta)
    mean = to_4d_tensor(bn_mean)
    var = to_4d_tensor(bn_var)
    
    # 3. Batch Normalization (using float64 math)
    x_norm = (x - mean) / torch.sqrt(var + eps)
    x_bn = gamma * x_norm + beta
    
    # 4. Activation Function (ReLU)
    x_relu = F.relu(x_bn)
    
    # 5. Convolution 2D
    out = F.conv2d(x_relu, conv_weight, bias=None, stride=1, padding=1)
    
    return out