from typing import List, Dict

class WordPieceTokenizer:
    def __init__(self, vocab: Dict[str, int], unk_token: str = "[UNK]", max_word_len: int = 100):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_word_len = max_word_len

    def _tokenize_word(self, word: str) -> List[str]:
        if len(word) > self.max_word_len:
            return [self.unk_token]

        output_tokens = []
        start = 0
        while start < len(word):
            end = len(word)
            cur_substr = None
            while start < end:
                substr = word[start:end]
                if start > 0:
                    substr = "##" + substr
                
                if substr in self.vocab:
                    cur_substr = substr
                    break
                end -= 1
            
            if cur_substr is None:
                return [self.unk_token]
            
            output_tokens.append(cur_substr)
            start = end
            
        return output_tokens # Ensure this is always reached if no error occurs

    def tokenize(self, text: str) -> List[str]:
        text = text.lower()
        result = []
        for word in text.strip().split():
            word_tokens = self._tokenize_word(word)
            # If _tokenize_word returned None, the next line would crash
            result.extend(word_tokens)
        return result

# Testing with your specific case:
vocab = {"a":15,"b":16,"c":17,"h":22,"go":32,"on":6,"un":0,"##a":18,"##b":19,"##c":20,"##d":34,"##e":23,"##l":24,"##n":27,"##o":33,"##p":25,"##s":8,"cat":4,"mat":7,"run":26,"sat":5,"the":3,"##ed":11,"##er":28,"##ly":31,"fast":29,"help":12,"play":9,"slow":30,"##ful":13,"##ing":10,"[UNK]":21,"##able":2,"##ness":14,"##believ":1}
tokenizer = WordPieceTokenizer(vocab)
print(tokenizer.tokenize("unbelievable")) 
# Output: ['un', '##believ', '##able']
