import numpy as np

def tanh(x):
    return np.tanh(x)

class BertPooler:
    """
    BERT Pooler: Extracts [CLS] and applies dense + tanh.
    """
    
    def __init__(self, hidden_size: int):
        self.hidden_size = hidden_size
        self.W = np.random.randn(hidden_size, hidden_size) * 0.02
        self.b = np.zeros(hidden_size)
    
    def forward(self, hidden_states: np.ndarray) -> np.ndarray:
        # 1. Extract [CLS] token (position 0) → shape (batch, hidden_size)
        cls_token = hidden_states[:, 0, :]
        
        # 2. Linear projection: cls @ W + b
        projected = cls_token @ self.W + self.b
        
        # 3. Tanh activation → output bounded to [-1, 1]
        return tanh(projected)


class SequenceClassifier:
    """
    Sequence classification head on top of BERT.
    """
    
    def __init__(self, hidden_size: int, num_classes: int):
        self.pooler = BertPooler(hidden_size)
        self.classifier = np.random.randn(hidden_size, num_classes) * 0.02
    
    def forward(self, hidden_states: np.ndarray) -> np.ndarray:
        # 1. Get pooled [CLS] representation → shape (batch, hidden_size)
        pooled_output = self.pooler.forward(hidden_states)
        
        # 2. Classification logits: pooled @ classifier → shape (batch, num_classes)
        return pooled_output @ self.classifier