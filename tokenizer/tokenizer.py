import regex as re
import json
import os

DATA_PATH = 'data/poems.txt'
TOKENIZER_PATH = 'models/bpe-token.json'

# Check it later
word_format = re.compile( r"""[\p{L}\p{M}]+(?:-[\p{L}\p{M}]+)* | \p{N}+ | § | [^\s\p{L}\p{N}]+ | [।॥] | \s+(?!\S) | \s+ """, re.VERBOSE)

def get_data(file):
    """ Read train data from file """
    
    with open(file, 'r', encoding='utf-8') as f:
        text = f.read()
    text = "§" + text.replace("\n\n", "§") + "§"
    print(f"\n",f"Text Length : {len(text)} ")
    return text


class Tokenizer:

    def __init__(self, tokenizer_path = None):

        self.merges = {} # { (p0, p1) : idx }   | {(int, int) : int}
        self.vocab = {}  # { token_id : bytes } | ( int : bytes)
        self.path = tokenizer_path
        if self.path and os.path.exists(self.path):
            print(f"\n Loading merges - tokenizer from {self.path} \n")
            self.load()
        else:
            self.vocab = {idx: bytes([idx]) for idx  in range(256)}


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
        """ Save merges in json format """
        merges_serialization = {
            f"{p0},{p1}": idx for (p0, p1), idx in self.merges.items()
        }
        with open(self.path, "w") as f:
            json.dump(merges_serialization, f)


    def load(self):
        """ Load merges and  rebuild vocab """
        with open(self.path, "r") as f:
            merges = json.load(f)
        self.merges = {
            tuple(map(int, k.split(","))) : v for  k, v in merges.items()
        }
        # rebuild vocab
        self.vocab = {idx: bytes([idx]) for idx  in range(256)} # (token_id -> bytes)
        for (p0, p1), idx in self.merges.items():
           self.vocab[idx] = self.vocab[p0] + self.vocab[p1] # addition '+' of two bytes object is concatenation


    def train(self, text : str, vocab_size = 300):
        """ Byte Pair Encoding """

        print('\n','-'*10, 'Training Tokenizer', '-'*10,'\n')
        self.merges = {}
        # tokens = list(text.encode('utf-8'))
        chunks = re.findall(word_format, text)
        token_chunks = [list(chunk.encode('utf-8')) for chunk in chunks]
        num_merges = vocab_size - 256
        next_id = 256 
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
            if i == int(0.25 * num_merges):
                print('-----25% done-----')
            elif i == int(0.5 * num_merges):
                print('-----50% done-----')
            elif i == int(0.75 * num_merges):
                print('-----75% done-----')
            elif i == num_merges - 1:
                print('-----100% done----- \n')
        tokens = [t for chunk in token_chunks for t in chunk]
        self.vocab = {idx: bytes([idx]) for idx  in range(256)} # (token_id -> bytes)
        for (p0, p1), idx in self.merges.items():
           self.vocab[idx] = self.vocab[p0] + self.vocab[p1] # addition '+' of two bytes object is concatenation
        self.save()
        return tokens
    

    def  decode(self, tokens):
        """given tokens (list of integers), return string"""

        # get (utf-8) byte tokens from vocabulary
        tokens = b"".join(self.vocab[idx] for idx in tokens)
        # decode (utf-8) bytes back into text string 
        text = tokens.decode("utf-8", errors='replace')
        return text

    def encode(self, text):
        """given text (str) , return corresponding tokens """

        # encode text string into (utf-8) bytes
        chunks = re.findall(word_format, text)
        tokens = []
        for chunk in chunks:
            chunk_tokens = list(chunk.encode('utf-8'))
            # apply merges to THIS chunk only, then extend
            tokens.extend(self._merge_chunk(chunk_tokens))
        return tokens

    

def main():

    # get the tokenizer
    tokenizer = Tokenizer(TOKENIZER_PATH)

    # training data
    text = get_data(DATA_PATH)

    # training loop
    # tokenizer.train(text, vocab_size=350)


    enc_text = tokenizer.encode(text)
    print(f'Tokens Length : {len(enc_text)}')
    print(f'Compression Ratio: {len(text.encode('utf-8')) / len(enc_text)}')


    sample = "एक छोटी कविता।"
    tokens = tokenizer.encode(sample)
    print(tokens)
    decoded = tokenizer.decode(tokens)
    print(decoded)
    print(decoded == sample)  # Should be True



if __name__ == "__main__":
    main()



