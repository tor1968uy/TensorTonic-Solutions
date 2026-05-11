import numpy as np

class BertEmbeddings:
    """
    BERT Embeddings = Token + Position + Segment
    """
    
    def __init__(self, vocab_size: int, max_position: int, hidden_size: int):
        self.hidden_size = hidden_size
        
        # Token embeddings: (vocab_size, hidden_size)
        self.token_embeddings = np.random.randn(vocab_size, hidden_size) * 0.02
        
        # Position embeddings (learned): (max_position, hidden_size)
        self.position_embeddings = np.random.randn(max_position, hidden_size) * 0.02
        
        # Segment embeddings (0 or 1): (2, hidden_size)
        self.segment_embeddings = np.random.randn(2, hidden_size) * 0.02
    
    def forward(self, token_ids: np.ndarray, segment_ids: np.ndarray) -> np.ndarray:
        """
        Returns: np.ndarray of shape (batch, seq_len, hidden_size) with combined embeddings
        """
        batch_size, seq_len = token_ids.shape
        
        # 1. Look up token embeddings
        # Result shape: (batch_size, seq_len, hidden_size)
        tokens_embedded = self.token_embeddings[token_ids]
        
        # 2. Look up segment embeddings
        # Result shape: (batch_size, seq_len, hidden_size)
        segments_embedded = self.segment_embeddings[segment_ids]
        
        # 3. Look up position embeddings
        # Create a range [0, 1, ..., seq_len-1]
        positions = np.arange(seq_len)
        # Result shape: (seq_len, hidden_size)
        positions_embedded = self.position_embeddings[positions]
        
        # 4. Sum them all together
        # NumPy broadcasting will handle adding (seq_len, hidden_size) 
        # to the (batch_size, seq_len, hidden_size) arrays automatically.
        embeddings = tokens_embedded + segments_embedded + positions_embedded
        
        return embeddings
