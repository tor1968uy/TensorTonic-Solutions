import numpy as np

def conv2d(x, W, b):
    x = np.asarray(x, dtype=float)
    W = np.asarray(W, dtype=float)
    b = np.asarray(b, dtype=float)
    
    # Renombrar para no pisar el array W
    N, C_in, H_in, W_in = x.shape
    C_out, _, kH, kW = W.shape
    
    H_out = H_in - kH + 1
    W_out = W_in - kW + 1
    
    out = np.zeros((N, C_out, H_out, W_out))
    
    for i in range(kH):
        for j in range(kW):
            x_slice = x[:, :, i:i+H_out, j:j+W_out]
            out += np.einsum('nchw,oc->nohw', x_slice, W[:, :, i, j])
            
    out += b.reshape(1, -1, 1, 1)
    
    return out

# Prueba
x_test = np.ones((1, 1, 3, 3))
W_test = np.ones((1, 1, 2, 2))
b_test = 0
print(conv2d(x_test, W_test, b_test))