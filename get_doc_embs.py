import numpy as np
import pickle
import torch
import torch.nn as nn
import copy
from sklearn.preprocessing import normalize
from torch.nn.functional import softmax


def combine_avg_doc_embs(sequences, model, idf_base):
    """
    Computes weighted average document embeddings from word embeddings and idf scores
    :param sequences: input documents
    :param model: either a simple embedding matrix or a trained model from which to extract the embedding layer
    :param idf_base: idf scores based on "above_threshold", "below_threshold" or "all"
    :return: the normalized, weighted averaged document embeddings
    """

    with open("public_data/vocab/bm25_word2weight_%s.pkl" %idf_base, 'rb') as infile:
        word2weight = pickle.load(infile)

    with open("public_data/vocab/word2id.pkl", 'rb') as infile:
        word2id = pickle.load(infile)
    id2word = {id: word for word, id in word2id.items()}

    if type(model) == str: #"8_gan" without additional classifier
        emb_path = "public_data/embeddings/word_embeddings_%s.pkl" %model
        with open(emb_path, 'rb') as infile:
            emb_matrix = np.asarray(pickle.load(infile), dtype="float32")

    else: #model, with classifier
        if hasattr(model, "encoder"):
            emb_matrix = model.encoder.state_dict()["embeddings.weight"]
        else:
            emb_matrix = model.state_dict()["embeddings.weight"]

        emb_matrix = np.asarray(emb_matrix.cpu().numpy(), dtype="float32")

    all_doc_embs = []
    for batch_id, seqs in enumerate(sequences):
        seqs = seqs.detach().numpy() #batch x seqlen
        word_embs = np.array([[emb_matrix[id].astype("float32") * word2weight.get(id2word[id], 1.0) for
                 id in seq] for seq in seqs]) #batchsize x seqlen x embdim

        for i in range(len(word_embs)):
            row = word_embs[i]
            row[np.isnan(row)] = 0.0

        doc_embs = np.true_divide(word_embs.sum(axis=1), (word_embs != 0.0).sum(axis=1)).astype("float32") #emb_dim
        all_doc_embs.append(doc_embs)

    all_doc_embs = np.concatenate(all_doc_embs, axis=0)

    for i in range(len(all_doc_embs)):
        row = all_doc_embs[i]
        row[np.isnan(row)] = 0.0

    return normalize(all_doc_embs, axis=1) #batchsize x emb_dim


def extract_from_model(doc_repr, sequences, model):
    """
    Extract document representations (last hidden states, linear scores, softmax scores)
    from model during or after training
    :param doc_repr: "hidden", "linear", "softmax"
    :param sequences: dataloader of input documents
    :param model: complete model
    :return: the normalized document vectors
    """

    if hasattr(model, "encoder"):
        model = model.encoder
    model = copy.deepcopy(model)

    if torch.cuda.is_available():
        dtype = torch.cuda.LongTensor
        device = torch.device('cuda')
    else:
        dtype = torch.LongTensor
        device = torch.device('cpu')
    model.to(device)
    model.eval()

    try:
        children = model.module.children()
    except AttributeError:
        children = model.children()
    linear = False
    for layer in children:
        if isinstance(layer, nn.Linear):
            linear = True

    with torch.no_grad():
        all_doc_embs = []
        for batch_id, seqs in enumerate(sequences):

            if linear:
                last_hidden, linear_scores = model(seqs.type(dtype))

                if doc_repr == "linear":
                    doc_embs = linear_scores.detach().cpu().numpy()

                elif doc_repr == "softmax":
                    doc_embs = softmax(linear_scores, dim=1).detach().cpu().numpy()

                elif doc_repr == "hidden":
                    doc_embs = last_hidden.detach().cpu().numpy()

            else:
                last_hidden = model(seqs.type(dtype))
                doc_embs = last_hidden.detach().cpu().numpy()

            all_doc_embs.append(doc_embs)

    all_doc_embs = np.concatenate(all_doc_embs, axis=0)

    for i in range(len(all_doc_embs)):
        row = all_doc_embs[i]
        row[np.isnan(row)] = 0.0

    return normalize(all_doc_embs, axis=1)







