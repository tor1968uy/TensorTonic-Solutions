import numpy as np

def relu(x):
    return np.maximum(0, x)

def batch_norm(x, gamma, beta, eps=1e-5):
    # Normalize across the batch dimension (axis 0)
    mean = np.mean(x, axis=0)
    # Use ddof=0 for population variance to match standard BN implementations
    var = np.var(x, axis=0)
    x_hat = (x - mean) / np.sqrt(var + eps)
    return gamma * x_hat + beta

def batch_norm_block(x, W1, W2, gamma1, beta1, gamma2, beta2, mode):
    x = np.array(x)
    identity = x
    
    if mode == "post":
        # Conv -> BN -> ReLU
        out = x @ W1
        out = batch_norm(out, gamma1, beta1)
        out = relu(out)
        
        # Conv -> BN
        out = out @ W2
        out = batch_norm(out, gamma2, beta2)
        
        # Residual + Final ReLU
        res = relu(out + identity)
        
    elif mode == "pre":
        # BN -> ReLU -> Conv
        out = batch_norm(x, gamma1, beta1)
        out = relu(out)
        out = out @ W1
        
        # BN -> ReLU -> Conv
        out = batch_norm(out, gamma2, beta2)
        out = relu(out)
        out = out @ W2
        
        # Pure Identity Skip (No final ReLU)
        res = out + identity
        
    # The test runner expects a dictionary with "output" and "mode"
    return {"output": res.tolist(), "mode": mode}