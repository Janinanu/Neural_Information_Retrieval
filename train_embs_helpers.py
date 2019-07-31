import pandas as pd
from custom_tokenizer import tokenize_and_clean

PARAMS = {"SIZE": 300, "WINDOW": 5, "MIN_COUNT": 5, "WORKERS": 10,
"SG": 1, "NEG": 5, "EPOCHS": 20}

def make_txt_file(file):
    """
    Turns documents from dataframe into txt file as input to unsupervised embedding training
    :param file: pickled dataframe
    """
    with open(file, 'rb') as infile:
        df = pd.read_pickle(infile, compression=None)
    texts = df["TEXT"].apply(lambda x: tokenize_and_clean(x))
    with open("public_data/inputs/data.txt", "w", encoding="utf-8") as out:
        for text in texts:
            out.write(" ".join(text) + "\n")

class YieldTexts:
    """
    Generator to yield training examples for Word2Vec training with gensim
    """
    def __init__(self, path):
        with open(path, "r") as infile:
            self.texts = infile.readlines()
            self.num = len(self.texts)
    def __iter__(self):
        for line in self.texts:
            tokens = line.split()
            yield tokens

if __name__ == "__main__":
    make_txt_file('public_data/inputs/dataset.pkl')

