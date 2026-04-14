import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words in sorted order.
        """
        # 1. Definir tokens especiales con sus IDs fijos
        special_tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        
        # 2. Recolectar palabras únicas de todos los textos
        unique_words = set()
        for text in texts:
            words = text.lower().split()
            unique_words.update(words)
        
        # 3. Ordenar las palabras alfabéticamente
        sorted_words = sorted(list(unique_words))
        
        # 4. Combinar y asignar IDs
        full_vocab = special_tokens + sorted_words
        
        self.word_to_id = {word: i for i, word in enumerate(full_vocab)}
        self.id_to_word = {i: word for i, word in enumerate(full_vocab)}
        self.vocab_size = len(full_vocab)
    
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK ID for unknown words.
        """
        words = text.lower().split()
        unk_id = self.word_to_id.get(self.unk_token, 1)
        
        # Mapear cada palabra a su ID, devolviendo el ID de <UNK> si no existe
        return [self.word_to_id.get(word, unk_id) for word in words]
    
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        Use <UNK> for unknown IDs.
        """
        unk_token = self.unk_token
        
        # Mapear cada ID a su palabra, devolviendo <UNK> si el ID no es reconocido
        decoded_words = [self.id_to_word.get(idx, unk_token) for idx in ids]
        
        # Unir con espacios
        return " ".join(decoded_words)