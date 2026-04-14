import numpy as np

def relu(x):
    """Standard ReLU activation function."""
    return np.maximum(0, x)

def resnet_forward(x, conv1, W1_b1, W2_b1, W1_b2, W2_b2, Ws_b2, fc):
    """
    Simplified ResNet-18 forward pass.
    x: Input features (batch, 3)
    conv1: First projection (3 -> 2)
    W1_b1, W2_b1: Weights for Block 1 (Identity skip)
    W1_b2, W2_b2: Weights for Block 2 (Main path)
    Ws_b2: Weights for Block 2 (Projection skip)
    fc: Final classification layer
    """
    # Ensure inputs are numpy arrays
    x = np.array(x)
    
    # --- Initial Layer ---
    # In ResNet, the first layer is usually a Conv + ReLU
    x = relu(x @ np.array(conv1))
    
    # --- Block 1: BasicBlock with Identity Shortcut ---
    # input_dim == output_dim (2 -> 2)
    identity1 = x
    
    # Main path
    out1 = relu(x @ np.array(W1_b1))
    out1 = out1 @ np.array(W2_b1)
    
    # Merge and ReLU
    x = relu(out1 + identity1)
    
    # --- Block 2: BasicBlock with Projection Shortcut ---
    # input_dim != output_dim (2 -> 3)
    # The shortcut must be projected to match the new feature dimension
    identity2 = x @ np.array(Ws_b2)
    
    # Main path
    out2 = relu(x @ np.array(W1_b2))
    out2 = out2 @ np.array(W2_b2)
    
    # Merge and ReLU
    x = relu(out2 + identity2)
    
    # --- Final FC Layer ---
    # Final classification logits
    logits = x @ np.array(fc)
    
    return logits