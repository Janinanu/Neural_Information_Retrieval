import numpy as np
import argparse
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from gensim.models.wrappers import FastText
from gensim.models import Word2Vec


def find_nns(experiment, keys):

    if experiment == "6":
        emb_model = Word2Vec.load("public_data/models/experiment_6/embeddings_6")

    elif experiment == "8":
        emb_model = FastText.load_fasttext_format("public_data/models/experiment_8/embeddings_8.bin")

    embeddings = [] #15*5 x 300
    words = [] #15*5
    for key in keys:
        for similar_word, _ in emb_model.most_similar(key, topn=5):
            words.append(similar_word)
            embeddings.append(emb_model[similar_word])

    return words, embeddings


def plot_nns(experiment, keys, words, embeddings):

    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(embeddings) #15*5 x 300 --> 15*5 x 2

    x_values = [x for x, _ in new_values] #15*5
    y_values = [y for _, y in new_values] #15*5

    xs = [x_values[x:x+5] for x in range(0, len(x_values), 5)] #15 x 5
    ys = [y_values[y:y+5] for y in range(0, len(y_values), 5)] #15 x 5

    colors = cm.rainbow(np.linspace(0, 1, len(keys)))
    plt.figure(figsize=(16, 16))
    for i in range(len(keys)):
        plt.scatter(xs[i], ys[i], c=colors[i], label=keys[i])
        for ii, word in enumerate(words):
            plt.annotate(words[ii],
                     xy=(x_values[ii], y_values[ii]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.legend(loc=4)
    plt.grid(True)
    plt.savefig("public_data/results/nn_%s.png" %experiment, dpi=80)
    plt.show()
    plt.close()


parser = argparse.ArgumentParser()
parser.add_argument('--experiment', type=str, help='which experiment is run, e.g. "8", "6"')
parser.add_argument('--k', type=int, nargs="?", default=6, help='k nearest neighbors, e.g. 5')
args = parser.parse_args()

keys = ["debt", "loan", "late", "purchase", "service", "frustrated",
       "car", "outdated", "call", "wrong", "issue", "respond", "student", "prepaid"]

words, embeddings = find_nns(experiment=args.experiment, keys=keys)
plot_nns(experiment=args.experiment, keys=keys, words=words, embeddings=embeddings)