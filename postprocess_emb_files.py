from gensim.models.keyedvectors import KeyedVectors
from gensim.models import Word2Vec


def bin_to_vec(bin_file, vec_file): #for 5
    model = KeyedVectors.load_word2vec_format(bin_file, binary=True)
    model.save_word2vec_format(vec_file, binary=False)

def gensim_to_vec(infile, outfile): #for 6
    w2v_model = Word2Vec.load(infile)
    w2v = w2v_model.wv
    with open(outfile, "w") as out:
        vocab_size = len(w2v.index2word)
        emb_dim = w2v.vector_size
        header = " ".join([str(vocab_size), str(emb_dim)])
        out.writelines(header + "\n")
        for word in w2v.index2word:
                vec = " ".join(str(x) for x in w2v[word].tolist())
                out_line = " ".join((word, vec))
                out.writelines(out_line + "\n")


if __name__ == "__main__":
    #gensim_to_vec(infile="public_data/models/experiment_6/embeddings_6",
                  #outfile="public_data/models/experiment_6/embeddings_6.vec")
    bin_to_vec(bin_file="public_data/models/experiment_5/GoogleNews-vectors-negative300.bin",
                vec_file="public_data/models/experiment_5/embeddings_5.vec")