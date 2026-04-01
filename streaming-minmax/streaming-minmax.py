import numpy as np

def streaming_minmax_init(D):
    return {
        'min': np.full(D, np.inf),
        'max': np.full(D, -np.inf)
    }

def streaming_minmax_update(state, X_batch, eps=1e-8):
    X_batch = np.asarray(X_batch, dtype=float)
    
    state['min'] = np.minimum(state['min'], X_batch.min(axis=0))
    state['max'] = np.maximum(state['max'], X_batch.max(axis=0))
    
    scale = state['max'] - state['min']
    normalized = (X_batch - state['min']) / (scale + eps)
    
    return normalized