import torch
import torch.nn.functional as F

def to_4d_tensor(param, device):
    """Utility helper to safely parse parameter lists into (1, C, 1, 1) float64 tensors."""
    if not isinstance(param, torch.Tensor):
        param = torch.tensor(param, dtype=torch.float64, device=device)
    else:
        param = param.to(dtype=torch.float64, device=device)
    return param.view(1, -1, 1, 1)

def composite_layer(x, bn_gamma, bn_beta, bn_mean, bn_var, conv_weight, eps):
    gamma = to_4d_tensor(bn_gamma, x.device)
    beta = to_4d_tensor(bn_beta, x.device)
    mean = to_4d_tensor(bn_mean, x.device)
    var = to_4d_tensor(bn_var, x.device)
    
    if not isinstance(conv_weight, torch.Tensor):
        conv_weight = torch.tensor(conv_weight, dtype=torch.float64, device=x.device)
    else:
        conv_weight = conv_weight.to(dtype=torch.float64, device=x.device)
        
    x_norm = (x - mean) / torch.sqrt(var + eps)
    x_bn = gamma * x_norm + beta
    x_relu = F.relu(x_bn)
    return F.conv2d(x_relu, conv_weight, bias=None, stride=1, padding=1)

def dense_block(x, layers, eps):
    features = [x]
    for layer_dict in layers:
        current_input = torch.cat(features, dim=1)
        layer_output = composite_layer(
            current_input, 
            layer_dict['bn_gamma'], layer_dict['bn_beta'], 
            layer_dict['bn_mean'], layer_dict['bn_var'], 
            layer_dict['conv_weight'], eps
        )
        features.append(layer_output)
    return torch.cat(features, dim=1)

def transition_layer(x, bn_gamma, bn_beta, bn_mean, bn_var, conv_weight, eps):
    gamma = to_4d_tensor(bn_gamma, x.device)
    beta = to_4d_tensor(bn_beta, x.device)
    mean = to_4d_tensor(bn_mean, x.device)
    var = to_4d_tensor(bn_var, x.device)
    
    if not isinstance(conv_weight, torch.Tensor):
        conv_weight = torch.tensor(conv_weight, dtype=torch.float64, device=x.device)
    else:
        conv_weight = conv_weight.to(dtype=torch.float64, device=x.device)
        
    x_norm = (x - mean) / torch.sqrt(var + eps)
    x_bn = gamma * x_norm + beta
    x_relu = F.relu(x_bn)
    x_conv = F.conv2d(x_relu, conv_weight, bias=None, stride=1, padding=0)
    return F.avg_pool2d(x_conv, kernel_size=2, stride=2)

def extract_final_bn_params(weights):
    """
    Locates and extracts the 4 required final BN parameters from weights 
    regardless of whether they are nested or flat-prefixed.
    """
    # Strategy A: Check common nested sub-dictionaries first
    for candidate in ["final_bn", "final_norm", "bn", "norm5"]:
        if candidate in weights and isinstance(weights[candidate], dict):
            d = weights[candidate]
            if "bn_gamma" in d or "gamma" in d:
                g = d.get("bn_gamma", d.get("gamma"))
                b = d.get("bn_beta", d.get("beta"))
                m = d.get("bn_mean", d.get("mean"))
                v = d.get("bn_var", d.get("var"))
                return g, b, m, v

    # Strategy B: Scan flat top-level keys for pattern matches
    g, b, m, v = None, None, None, None
    for k, val in weights.items():
        k_lower = k.lower()
        if "blocks" in k_lower or "transitions" in k_lower or "stem" in k_lower or "fc_" in k_lower:
            continue
        if "gamma" in k_lower:
            g = val
        elif "beta" in k_lower:
            b = val
        elif "mean" in k_lower:
            m = val
        elif "var" in k_lower:
            v = val

    if g is not None and b is not None and m is not None and v is not None:
        return g, b, m, v
        
    # Strategy C: Direct fallback to standard un-prefixed top level
    return weights.get("bn_gamma"), weights.get("bn_beta"), weights.get("bn_mean"), weights.get("bn_var")


def densenet_forward(x, weights, growth_rate, eps=1e-5):
    """
    Returns torch.Tensor of shape (N, num_classes) containing class logits.
    """
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float64)
    else:
        x = x.to(torch.float64)
        
    stem_weight = weights["stem_conv"]
    if not isinstance(stem_weight, torch.Tensor):
        stem_weight = torch.tensor(stem_weight, dtype=torch.float64, device=x.device)
    else:
        stem_weight = stem_weight.to(dtype=torch.float64, device=x.device)

    # 1. Stem Layer
    h = F.conv2d(x, stem_weight, bias=None, stride=1, padding=1)
    
    # 2. Sequential Network Blocks Loop
    blocks = weights["blocks"]
    transitions = weights["transitions"]
    num_blocks = len(blocks)
    
    for i in range(num_blocks):
        h = dense_block(h, blocks[i], eps)
        if i < num_blocks - 1:
            trans_dict = transitions[i]
            h = transition_layer(
                h, 
                trans_dict['bn_gamma'], trans_dict['bn_beta'], 
                trans_dict['bn_mean'], trans_dict['bn_var'], 
                trans_dict['conv_weight'], eps
            )
            
    # 3. Dynamic Extraction of Final BN Layer Parameters
    bn_gamma, bn_beta, bn_mean, bn_var = extract_final_bn_params(weights)

    final_gamma = to_4d_tensor(bn_gamma, h.device)
    final_beta = to_4d_tensor(bn_beta, h.device)
    final_mean = to_4d_tensor(bn_mean, h.device)
    final_var = to_4d_tensor(bn_var, h.device)
    
    # 4. Final Classification Pre-activation Sequence
    h_norm = (h - final_mean) / torch.sqrt(final_var + eps)
    h_bn = final_gamma * h_norm + final_beta
    h_relu = F.relu(h_bn)
    
    # 5. Global Average Pooling (GAP)
    H, W = h_relu.shape[2], h_relu.shape[3]
    pooled = h_relu.sum(dim=(2, 3)) / (H * W)
    
    # 6. Linear Projection Head
    fc_w = weights["fc_weight"]
    fc_b = weights["fc_bias"]
    
    if not isinstance(fc_w, torch.Tensor):
        fc_w = torch.tensor(fc_w, dtype=torch.float64, device=h.device)
    else:
        fc_w = fc_w.to(dtype=torch.float64, device=h.device)
        
    if not isinstance(fc_b, torch.Tensor):
        fc_b = torch.tensor(fc_b, dtype=torch.float64, device=h.device)
    else:
        fc_b = fc_b.to(dtype=torch.float64, device=h.device)
        
    logits = F.linear(pooled, fc_w, fc_b)
    return logits