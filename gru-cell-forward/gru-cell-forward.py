import numpy as np

def _sigmoid(x):
    """Numerically stable sigmoid function"""
    return np.where(x >= 0, 1.0/(1.0+np.exp(-x)), np.exp(x)/(1.0+np.exp(x)))

def _as2d(a, feat):
    """Convert 1D array to 2D and track if conversion happened"""
    a = np.asarray(a, dtype=float)
    if a.ndim == 1:
        return a.reshape(1, feat), True
    return a, False

def gru_cell_forward(x, h_prev, params):
    """
    Implement the GRU forward pass for one time step.
    """
    x_arr = np.asarray(x, dtype=float)
    h_arr = np.asarray(h_prev, dtype=float)
    
    d_feat = x_arr.shape[-1] if x_arr.size > 0 else 0
    h_feat = h_arr.shape[-1] if h_arr.size > 0 else 0

    x_2d, x_was_1d = _as2d(x_arr, d_feat)
    h_prev_2d, h_was_1d = _as2d(h_arr, h_feat)
    
    Wz, Uz, bz = np.array(params['Wz']), np.array(params['Uz']), np.array(params['bz'])
    Wr, Ur, br = np.array(params['Wr']), np.array(params['Ur']), np.array(params['br'])
    Wh, Uh, bh = np.array(params['Wh']), np.array(params['Uh']), np.array(params['bh'])

    # (N,D) @ (D,H) + (N,H) @ (H,H) + (H,) -> (N,H)
    z = _sigmoid(np.dot(x_2d, Wz) + np.dot(h_prev_2d, Uz) + bz)
    r = _sigmoid(np.dot(x_2d, Wr) + np.dot(h_prev_2d, Ur) + br)
    h_tilde = np.tanh(np.dot(x_2d, Wh) + np.dot(r * h_prev_2d, Uh) + bh)
    
    h_next = (1 - z) * h_prev_2d + z * h_tilde

    if x_was_1d or h_was_1d:
        return h_next.ravel()
    
    return h_next