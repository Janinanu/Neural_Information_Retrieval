import argparse
import numpy as np
import pickle
from nltk.corpus import stopwords
from gensim.models.wrappers import FastText
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from train_embs_helpers import PARAMS

def make_word2id():
    """
    Stores word2id dictionary from words in training vocabulary
    """
    with open("public_data/stats/stats_train.pkl", 'rb') as stats:
        stats = pickle.load(stats)
    vocab = stats["VOCAB"]
    word2id = {word: id for id, word in enumerate(["PAD"] + ["UNK"] + vocab)}
    with open('public_data/vocab/word2id.pkl', 'wb') as out:
        pickle.dump(word2id, out, protocol=4)


def prepare_word_emb_matrices(experiment):
    """
    Initializes word embeddings for each word in training vocabulary
    from pretrained or custom-trained embedding files
    :param experiment: the ID of the word embedding file
    :return: the training embedding matrix
    """

    with open("public_data/stats/stats_train.pkl", 'rb') as stats:
        stats = pickle.load(stats)
    vocab = stats["VOCAB"]
    stops = [word.lower() for word in set(stopwords.words('english'))]
    vocab = vocab + stops

    if experiment == "RANDOM":
        word_embs = np.random.uniform(low=-1.0, high=1.0, size=(len(vocab), PARAMS["SIZE"])).astype("float32")

    else:
        word_embs = []
        count_unk = 0
        count_kn = 0

        if experiment == "5":
            emb_model = KeyedVectors.load_word2vec_format("public_data/models/experiment_5/embeddings_5.bin",
                                                              binary=True)
        elif experiment == "6":
            emb_model = Word2Vec.load("public_data/models/experiment_6/embeddings_6")

        elif experiment in ["7", "8"]:
            emb_model = FastText.load_fasttext_format("public_data/models/experiment_%s/embeddings_%s.bin"
                                                      %(experiment, experiment))
        for word in vocab:
            if word in emb_model:
                word_embs.append(emb_model[word])
                count_kn += 1
            else:
                word_embs.append(np.random.uniform(low=-1.0, high=1.0, size=PARAMS["SIZE"]))
                count_unk += 1

        word_embs = np.array(word_embs).astype("float32")
        print(count_unk / (count_kn + count_unk))

    pad = np.zeros(shape=PARAMS["SIZE"]).astype("float32")
    unk = np.random.uniform(low=-1.0, high=1.0, size=PARAMS["SIZE"]).astype("float32")
    word_embs = np.insert(word_embs, 0, unk, axis=0) #id 1
    word_embs = np.insert(word_embs, 0, pad, axis=0) #id 0

    with open("public_data/embeddings/word_embeddings_%s.pkl" %experiment, 'wb') as out:
        pickle.dump(word_embs, out, protocol=4)

    return word_embs


parser = argparse.ArgumentParser()
parser.add_argument('--experiment', type=str, help='which word embeddings to use: "RANDOM", "5", "6", "7", "8"')
args = parser.parse_args()

if __name__ == "__main__":
    make_word2id()
    #prepare_word_emb_matrices(args.experiment)
