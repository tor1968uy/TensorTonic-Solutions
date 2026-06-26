import torch
import torch.nn.functional as F

def bottleneck_layer(x, bn1_gamma, bn1_beta, bn1_mean, bn1_var, conv1_weight,
                     bn2_gamma, bn2_beta, bn2_mean, bn2_var, conv2_weight, eps=1e-5):
    """
    Returns torch.Tensor of shape (N, growth_rate, H, W) after the two-stage bottleneck composite.
    Computes everything using torch.float64 precision.
    """
    # 1. Force float64 precision on structural inputs
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float64)
    else:
        x = x.to(torch.float64)
        
    if not isinstance(conv1_weight, torch.Tensor):
        conv1_weight = torch.tensor(conv1_weight, dtype=torch.float64)
    else:
        conv1_weight = conv1_weight.to(torch.float64)
        
    if not isinstance(conv2_weight, torch.Tensor):
        conv2_weight = torch.tensor(conv2_weight, dtype=torch.float64)
    else:
        conv2_weight = conv2_weight.to(torch.float64)

    # Helper function to parse 1D BN parameters into (1, C, 1, 1) float64 shapes
    def to_4d_tensor(param, device):
        if not isinstance(param, torch.Tensor):
            param = torch.tensor(param, dtype=torch.float64, device=device)
        else:
            param = param.to(dtype=torch.float64, device=device)
        return param.view(1, -1, 1, 1)

    # --- STAGE 1: BN1 -> ReLU -> Conv 1x1 ---
    g1 = to_4d_tensor(bn1_gamma, x.device)
    b1 = to_4d_tensor(bn1_beta, x.device)
    m1 = to_4d_tensor(bn1_mean, x.device)
    v1 = to_4d_tensor(bn1_var, x.device)
    
    x_norm1 = (x - m1) / torch.sqrt(v1 + eps)
    x_bn1 = g1 * x_norm1 + b1
    x_relu1 = F.relu(x_bn1)
    
    # 1x1 Convolution maps input channels C -> 4k (padding=0)
    y1 = F.conv2d(x_relu1, conv1_weight, bias=None, stride=1, padding=0)

    # --- STAGE 2: BN2 -> ReLU -> Conv 3x3 ---
    g2 = to_4d_tensor(bn2_gamma, y1.device)
    b2 = to_4d_tensor(bn2_beta, y1.device)
    m2 = to_4d_tensor(bn2_mean, y1.device)
    v2 = to_4d_tensor(bn2_var, y1.device)
    
    y1_norm = (y1 - m2) / torch.sqrt(v2 + eps)
    y1_bn = g2 * y1_norm + b2
    y1_relu = F.relu(y1_bn)
    
    # 3x3 Convolution maps 4k channels -> k channels (padding=1 keeps spatial sizes intact)
    y2 = F.conv2d(y1_relu, conv2_weight, bias=None, stride=1, padding=1)

    return y2