import numpy as np
from typing import List

class MockBertEncoder:
    """Simulated BERT encoder with configurable layers."""
    
    def __init__(self, hidden_size: int, num_layers: int):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.layers = [np.random.randn(hidden_size, hidden_size) * 0.01 for _ in range(num_layers)]
        self.layer_frozen = [False] * num_layers
    
    def freeze_layers(self, layer_indices: List[int]):
        for idx in layer_indices:
            self.layer_frozen[idx] = True
    
    def unfreeze_all(self):
        self.layer_frozen = [False] * self.num_layers
    
    def forward(self, embeddings: np.ndarray) -> np.ndarray:
        """Forward pass: x = x @ layer_W + x for each layer (residual connection)."""
        x = embeddings
        for i, layer in enumerate(self.layers):
            x = x @ layer + x
        return x


class BertForSequenceClassification:
    """BERT with sequence classification head (uses [CLS] token)."""
    
    def __init__(self, hidden_size: int, num_labels: int, num_layers: int = 3):
        self.encoder = MockBertEncoder(hidden_size, num_layers)
        self.classifier = np.random.randn(hidden_size, num_labels) * 0.02
    
    def forward(self, embeddings: np.ndarray) -> np.ndarray:
        # 1. Pass through encoder → (batch, seq_len, hidden_size)
        hidden = self.encoder.forward(embeddings)
        
        # 2. Extract [CLS] token at position 0 → (batch, hidden_size)
        cls_token = hidden[:, 0, :]
        
        # 3. Linear classification head → (batch, num_labels)
        return cls_token @ self.classifier


class BertForTokenClassification:
    """BERT with token-level classification head (NER, POS)."""
    
    def __init__(self, hidden_size: int, num_labels: int, num_layers: int = 3):
        self.encoder = MockBertEncoder(hidden_size, num_layers)
        self.classifier = np.random.randn(hidden_size, num_labels) * 0.02
    
    def forward(self, embeddings: np.ndarray) -> np.ndarray:
        # 1. Pass through encoder → (batch, seq_len, hidden_size)
        hidden = self.encoder.forward(embeddings)
        
        # 2. Classify ALL tokens → (batch, seq_len, num_labels)
        # @ broadcasts correctly over batch and seq_len dimensions
        return hidden @ self.classifier