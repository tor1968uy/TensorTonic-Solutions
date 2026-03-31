import numpy as np

def stratified_split(X, y, test_size=0.2, rng=None):
    X = np.array(X)
    y = np.array(y)
    
    classes = np.unique(y)
    train_indices = []
    test_indices = []
    
    # Si no se provee rng, usar default_rng sin semilla fija
    if rng is None:
        rng = np.random.default_rng()
    
    for cls in classes:
        cls_idx = np.where(y == cls)[0]
        rng.shuffle(cls_idx)
        
        n_class = len(cls_idx)
        n_test = int(round(n_class * test_size))
        n_test = min(max(n_test, 0), n_class - 1)
        
        # Test toma los PRIMEROS n_test (después del shuffle)
        tst = cls_idx[:n_test]
        trn = cls_idx[n_test:]
        
        test_indices.extend(tst)
        train_indices.extend(trn)
    
    train_idx_final = np.sort(np.array(train_indices))
    test_idx_final  = np.sort(np.array(test_indices))
    
    return X[train_idx_final], X[test_idx_final], y[train_idx_final], y[test_idx_final]