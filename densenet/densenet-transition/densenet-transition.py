import torch
import torch.nn.functional as F

def transition_layer(x, bn_gamma, bn_beta, bn_mean, bn_var, conv_weight, eps=1e-5):
    """
    Returns torch.Tensor of shape (N, out_channels, H//2, W//2) 
    after BN-ReLU-1x1Conv then 2x2 average pooling.
    Evaluated using high-precision torch.float64 math.
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

    # 2. Convert and reshape BN parameters to (1, C, 1, 1) for broadcasting
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
    
    # 3. Batch Normalization & Pre-activation ReLU
    x_norm = (x - mean) / torch.sqrt(var + eps)
    x_bn = gamma * x_norm + beta
    x_relu = F.relu(x_bn)
    
    # 4. 1x1 Convolution (padding=0 compress channels from C to C_out)
    x_conv = F.conv2d(x_relu, conv_weight, bias=None, stride=1, padding=0)
    
    # 5. Downsampling via 2x2 Average Pooling with stride 2
    out = F.avg_pool2d(x_conv, kernel_size=2, stride=2)
    
    return out