import regex as re
import json
import os
import base64

DATA_PATH = 'data/poems.txt'
TOKENIZER_PATH = 'models/merges.json'
VOCABULARY_PATH = 'models/vocab.json'
VOCAB_SIZE=1024

# Check it later
word_format = re.compile( r"""[\p{L}\p{M}]+(?:-[\p{L}\p{M}]+)* | \p{N}+ | § | [^\s\p{L}\p{N}]+ | [।॥] | \s+(?!\S) | \s+ """, re.VERBOSE)

SPECIAL_TOKENS = {
    '<eos>': 256,
    '<poem>': 257,
    '<pad>': 258,
}


class Tokenizer:

    def __init__(self, tokenizer_path = None, vocabulary_path = None):

        self.merges = {} # { (p0, p1) : idx }   | {(int, int) : int}
        self.vocab = {}  # { token_id : bytes } | ( int : bytes)
        self.tokenizer_path = tokenizer_path
        self.vocab_path = vocabulary_path
        self.special_tokens = SPECIAL_TOKENS
        self.special_token_ids = {v: k for k, v in SPECIAL_TOKENS.items()}  # {256: '<eos>', ...}


        if self.tokenizer_path and os.path.exists(self.tokenizer_path):
            print(f"\n Loading Tokenizer \n")
            self.load()
        else:
            self.vocab = {idx: bytes([idx]) for idx  in range(256)} # (token_id -> bytes)
            for token, idx in self.special_tokens.items():
                self.vocab[idx] = token.encode('utf-8')


    def get_pairs(self, tokens: list)  -> dict:
        """ Get the { (pair_0, pair_1) : counts} from a list of tokens"""
        counts = {}
        for pair in zip(tokens, tokens[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts


    def merge (self, ids: list, pair: tuple, idx: int) -> list:
        """ Function  to merge a pair 'tuple' into sinlge number 'idx' in a tokens list"""
        newids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                newids.append(idx)
                i  += 2
            else:
                newids.append(ids[i])
                i +=  1
        return newids
    
    def _merge_chunk(self, chunk_tokens: list) -> list:
        """Apply merges to single chunk only"""
        while True:
            pairs = self.get_pairs(chunk_tokens)
            valid_pairs = [p for p in pairs if p in self.merges]
            if not valid_pairs:
                break
            pair = min(valid_pairs, key=lambda p: self.merges[p])
            idx = self.merges[pair]
            chunk_tokens = self.merge(chunk_tokens, pair, idx)
        return chunk_tokens


    def save(self):
        """ Save merges and vocabulary in json format """
        merges_serialization = {
            f"{p0},{p1}": idx for (p0, p1), idx in self.merges.items()
        }
        # vocabulary = self.vocab
        vocabulary = {
            base64.b64encode(b).decode('ascii') : idx for idx, b in self.vocab.items()
        }
        special = self.special_tokens   

        with open(self.tokenizer_path, "w") as f:
            json.dump({'merges' : merges_serialization, 'special_tokens': special}, f)
        with open(self.vocab_path, "w") as f:
            json.dump(vocabulary, f)
        


    def load(self):
        with open(self.tokenizer_path, "r") as f:
            data = json.load(f)
        
        # handle both old format (just merges) and new format
        if isinstance(data, dict) and 'merges' in data:
            merges = data['merges']
            self.special_tokens = data.get('special_tokens', SPECIAL_TOKENS)
        else:
            merges = data  # old format fallback
            self.special_tokens = SPECIAL_TOKENS

        with open(self.vocab_path, 'r') as f:
            vocab = json.load(f)

        self.special_token_ids = {v: k for k, v in self.special_tokens.items()}
        self.vocab = {v: base64.b64decode(k) for k, v in vocab.items()}
        self.merges = {tuple(map(int, k.split(','))): v for k, v in merges.items()}


    def train(self, text : str, vocab_size = 300):
        """ Byte Pair Encoding """

        print('\n','-'*10, 'Training Tokenizer', '-'*10,'\n')
        self.merges = {}
        # reset vocab to base + special tokens
        # self.vocab = {idx: bytes([idx]) for idx in range(256)}
        # for token, idx

        chunks = re.findall(word_format, text)
        token_chunks = [list(chunk.encode('utf-8')) for chunk in chunks]

        first_merge_id = 256 + len(self.special_tokens)
        num_merges = vocab_size - first_merge_id
        next_id = first_merge_id
    
        for i in range(num_merges):
            # count pairs WITHIN each chunk only
            pairs = {}
            for chunk_tokens  in token_chunks:
                for pair, count in self.get_pairs(chunk_tokens).items():
                    pairs[pair] = pairs.get(pair, 0) + count
            if not pairs or max(pairs.values()) < 2:
                break
            pair = max(pairs, key=pairs.get)
            # merge within each chunk separately
            token_chunks = [self.merge(chunk, pair, next_id) for chunk in token_chunks]
            self.merges[pair] = next_id
            next_id += 1

            if i == int(0.1 * num_merges):
                print('-----10% done-----')
            elif i == int(0.2 * num_merges):
                print('-----20% done-----')
            elif i == int(0.3 * num_merges):
                print('-----30% done-----')
            elif i == int(0.4 * num_merges):
                print('-----40% done-----')
            elif i == int(0.5 * num_merges):
                print('-----50% done-----')
            elif i == int(0.6 * num_merges):
                print('-----60% done-----')
            elif i == int(0.7 * num_merges):
                print('-----70% done-----')
            elif i == int(0.8 * num_merges):
                print('-----80% done-----')
            elif i == int(0.9 * num_merges):
                print('-----90% done-----')
            elif i == num_merges - 1:
                print('-----100% done----- \n')
        
        tokens = [t for chunk in token_chunks for t in chunk]
        for (p0, p1), idx in self.merges.items():
           self.vocab[idx] = self.vocab[p0] + self.vocab[p1] # addition '+' of two bytes object is concatenation
        self.save()
        return tokens
    

    def  decode(self, tokens):
        """given tokens (list of integers), return string"""

        parts = []
        for idx in tokens:
            if idx in self.special_token_ids:
                parts.append(self.special_token_ids[idx].encode('utf-8'))
            else:
                parts.append(self.vocab[idx])
        text = b"".join(parts).decode('utf-8', errors='replace')
        return text


       

    def encode(self, text):
        """given text (str) , return corresponding tokens """

        import re as stdlib_re
        special_pattern = '('+ '|'.join(
            stdlib_re.escape(tok) for tok in sorted(self.special_tokens, key=len, reverse=True)
        )+')'
        parts = stdlib_re.split(special_pattern, text)

        tokens = []
        for part in parts:
            if  part in self.special_tokens:
                tokens.append(self.special_tokens[part])
            elif part:
                chunks = re.findall(word_format, part)
                for chunk in chunks:
                    chunk_tokens= list(chunk.encode('utf-8'))
                    tokens.extend(self._merge_chunk(chunk_tokens))
        return tokens

    

def main():

    tokenizer = Tokenizer(TOKENIZER_PATH, VOCABULARY_PATH)

    sample = "एक छोटी कविता।"
    tokens = tokenizer.encode(sample)
    print(tokens)
    decoded = tokenizer.decode(tokens)
    print(decoded)
    print(decoded == sample)  # Should be True


if __name__ == "__main__":
    main()



