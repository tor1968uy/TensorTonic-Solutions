import numpy as np

def adadelta_step(w, grad, E_grad_sq, E_update_sq, rho=0.9, eps=1e-6):
    w          = np.asarray(w,          dtype=float)
    grad       = np.asarray(grad,       dtype=float)
    E_grad_sq  = np.asarray(E_grad_sq,  dtype=float)
    E_update_sq= np.asarray(E_update_sq,dtype=float)

    # Actualizar media móvil de gradientes al cuadrado
    E_grad_sq_new = rho * E_grad_sq + (1 - rho) * grad**2

    # Calcular delta_w usando ratio RMS(update) / RMS(grad)
    rms_update = np.sqrt(E_update_sq + eps)
    rms_grad   = np.sqrt(E_grad_sq_new + eps)
    delta_w    = -(rms_update / rms_grad) * grad

    # Actualizar media móvil de updates al cuadrado
    E_update_sq_new = rho * E_update_sq + (1 - rho) * delta_w**2

    w_new = w + delta_w

    return w_new, E_grad_sq_new, E_update_sq_new