import numpy as np

def detect_skew(train_dist, serving_dist, threshold=0.2, eps=1e-10):

    def psi_single(train, serving):
        t = np.array(train, dtype=np.float64)
        s = np.array(serving, dtype=np.float64)
        # No normalizar — asumir que ya suman 1
        t_e = t + eps
        s_e = s + eps
        return float(np.sum((s_e - t_e) * np.log(s_e / t_e)))

    if isinstance(train_dist, dict):
        result = {}
        for feature in train_dist:
            psi = psi_single(train_dist[feature], serving_dist[feature])
            result[feature] = {'psi': psi, 'skewed': bool(psi >= threshold)}
        return result
    else:
        psi = psi_single(train_dist, serving_dist)
        return {'psi': psi, 'skewed': bool(psi >= threshold)}