import fasttext
from gensim.models import Word2Vec
from train_embs_helpers import PARAMS, YieldTexts

def train_fasttext():
    """
    Trains Fasttext model from txt file and stores embedding file
    :return: trained model
    """
    infile = "public_data/inputs/data.txt"
    outfile = 'public_data/models/experiment_8/embeddings_8'

    model = fasttext.skipgram(infile, outfile,
                              dim=PARAMS["SIZE"],
                              ws=PARAMS["WINDOW"],
                              min_count=PARAMS["MIN_COUNT"],
                              neg=PARAMS["NEG"],
                              epoch=PARAMS["EPOCHS"])

    return model


def train_word2vec():
    """
    Trains Word2Vec model from txt file and stores embedding file
    :return: trained model
    """

    infile = "public_data/inputs/data.txt"
    outfile = 'public_data/models/experiment_6/embeddings_6'

    texts = YieldTexts(infile)
    model = Word2Vec(
        size=PARAMS["SIZE"],
        window=PARAMS["WINDOW"],
        min_count=PARAMS["MIN_COUNT"],
        workers=PARAMS["WORKERS"],
        sg=PARAMS["SG"],
        negative=PARAMS["NEG"])
    model.build_vocab(texts)
    model.train(texts, total_examples=texts.num, epochs=PARAMS["EPOCHS"])
    model.save(outfile)

    return model

if __name__ == "__main__":
    model = train_fasttext()
    #model = train_word2vec()

