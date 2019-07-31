import faiss
from scipy.stats import entropy
from gensim.summarization.bm25 import BM25
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
import numpy as np
from scipy.sparse import csr_matrix
from collections import defaultdict
import pickle
import os
from custom_tokenizer import tokenize_and_clean


def build_index(collection):
    """
    Builds Faiss index from documents in collection, using inner product as distance metric
    (will be equal to cosine similarity because collection vectors are normalized before)
    :param collection: normalized vectors of known documents
    :return: index of collection
    """

    index = faiss.IndexFlatIP(collection.shape[1])
    index.add(collection)

    return index

def search_index(index, k, queries):
    """
    Searches collection index using Faiss, ranks documents in collection by cosine similarity
    for each of the queries
    :param index: index of collection documents
    :param k: number of documents to be ranked for each query
    :param queries: all query vectors
    :return: top_ids: vector of dimension (number of queries, collection size) along with the
    respective similarity scores
    """

    top_scores, top_ids = index.search(queries, k)

    return top_ids, top_scores

def jsd(k, queries, collection):
    """
    Computes Jensen-Shannon divergence between each query and the collection
    :param k: The number of documents in the collection to be ranked
    :param queries: Softmax score vectors of the queries
    :param collection: Softmax score vectors of the collection
    :return: top_ids: vector of dimension (number of queries, collection size) along with the
    respective similarity scores
    """

    top_ids = []
    for i, p in enumerate(queries):
        jsds = []
        for q in collection:
            m = 0.5*(p+q)
            kld_pm = entropy(pk=p, qk=m)
            kld_qm = entropy(pk=q, qk=m)
            jsd_pq = 0.5*kld_pm + 0.5*kld_qm
            jsds.append(jsd_pq)
        if k != None:
            ids = np.argpartition(jsds, k)[:k]
        else:
            ids = np.argsort(np.array(jsds))
        top_ids.append(ids)
    top_ids = np.array(top_ids)

    return top_ids

def bm25(k, queries, collection, idf_base):
    """
    Computes BM25 scores for each query and the documents in the collection
    :param k: The number of documents in the collection to be ranked
    :param queries: all query vectors
    :param collection: all collection vectors
    :param idf_base: store idf with identifier "above_threshold", "below_threshold", "all"
    :return:
    """

    bm25 = BM25(collection)

    if not os.path.exists("public_data/vocab/bm25_word2weight_%s.pkl" %idf_base):
        with open("public_data/vocab/bm25_word2weight_%s.pkl" %idf_base, 'wb') as out:
            pickle.dump(bm25.idf, out, protocol=4)

    avg_idf = sum(bm25.idf.values()) / len(bm25.idf.values())
    top_ids = []
    for i, query in enumerate(queries):
        scores = bm25.get_scores(query, avg_idf)
        ids = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        top_ids.append(ids)
    top_ids = np.array(top_ids)

    return top_ids

def dummy_fun(doc):
    return doc #lambda x: x

def compute_tfidf(collection, queries, idf_base):
    """
    Computes tfidf vectors
    :param collection: all collection documents
    :param queries: all query documents
    :param idf_base: store idf with identifier "above_threshold", "below_threshold", "all"
    :return: document vectors for documents in collection and queries
    """

    vectorizer = TfidfVectorizer(
        analyzer='word',
        tokenizer=dummy_fun,
        preprocessor=dummy_fun,
        token_pattern=None)

    texts = list(collection["TEXT"].apply(lambda x: tokenize_and_clean(x)))
    collection = vectorizer.fit_transform(texts)
    collection = normalize(csr_matrix(collection, dtype=np.float32).toarray(), copy=True)

    texts = list(queries["TEXT"].apply(lambda x: tokenize_and_clean(x)))
    queries = vectorizer.transform(texts)
    queries = normalize(csr_matrix(queries, dtype=np.float32).toarray(), copy=True)

    max_idf = max(vectorizer.idf_)
    word2weight = defaultdict(lambda: max_idf, [(w, vectorizer.idf_[i]) for w, i in vectorizer.vocabulary_.items()])
    with open("public_data/vocab/tf_idf_word2weight_%s.pkl" %idf_base, 'wb') as out:
        pickle.dump(dict(word2weight), out, protocol=4)

    return collection, queries