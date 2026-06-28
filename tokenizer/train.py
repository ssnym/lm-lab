from .bpe import Tokenizer, word_format

DATA_PATH = 'data/pretrain_train.txt'
TOKENIZER_PATH = 'models/merges.json'
VOCABULARY_PATH = 'models/vocab.json'
VOCAB_SIZE=1024


def get_data(file):
    """ Read train data from file """
    
    with open(file, 'r', encoding='utf-8') as f:
        text = f.read()
    text = "§" + text.replace("\n\n", "§") + "§"
    print(f"\n",f"Text Length : {len(text)} ")
    return text


def train():

    # get the tokenizer
    tokenizer = Tokenizer(TOKENIZER_PATH, VOCABULARY_PATH)

    # training data
    text = get_data(DATA_PATH)

    # training loop
    tokenizer.train(text, VOCAB_SIZE)

    enc_text = tokenizer.encode(text)
    print(f'Tokens Length : {len(enc_text)}')
    print(f'Compression Ratio: {len(text.encode('utf-8')) / len(enc_text)}')

    print(f"\n", "-"*10, " Tokenizer Training Done ", "-"*10, "\n")


if __name__ == "__main__":
    train()
