import argparse
import pandas as pd
import datetime
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.preprocessing import normalize
from custom_tokenizer import tokenize_and_clean
from get_doc_embs import combine_avg_doc_embs, extract_from_model
from sim_search import search_index, build_index, jsd, bm25, compute_tfidf
from metrics import compute_mean_avg_prec
from classifier_helpers import load_model_and_configuration, \
    IR_Dataset, MulticlassDataset, BinaryPairDataset, BinaryTripletDataset
from test_classifiers import test_multiclass_model, test_pair_cel_model, test_triplet_model

class Evaluation_IR:

    """Evaluates the MAP of the baselines,
    of the pre-classifier weighted averaged word embeddings,
    and of the different document representations extracted from the classifiers,
    optionally evaluates the test loss and test accuracy of the classifiers"""
    def __init__(self, experiment, label_type, doc_repr, which_labels, threshold, test_acc_loss):

        self.experiment = experiment
        self.threshold = threshold
        self.which_labels = which_labels
        self. test_acc_loss = test_acc_loss

        if self.experiment in ["BM25", "TF-IDF"]:
            self.doc_repr = None
            self.label_type = label_type
            self.save_as = "_".join([self.experiment, self.label_type, which_labels, str(threshold)])

        #word embeddings & doc representations from trained classifiers
        elif "mucl" in self.experiment or "pair" in self.experiment or "triplet" in self.experiment:

            if "pair" in self.experiment or "triplet" in self.experiment:
                assert doc_repr in ["avg", "hidden"]
            else:
                assert doc_repr in ["avg", "hidden", "linear", "softmax"]
            self.doc_repr = doc_repr

            if "mucl" in self.experiment:
                self.model, self.label_type, self.label2id, _ = load_model_and_configuration(self.experiment)

            elif "pair_cel" in self.experiment:
                self.model, self.label_type, self.threshold = load_model_and_configuration(self.experiment)

            elif "pair" in self.experiment or "triplet" in self.experiment:
                self.model, self.label_type, self.threshold, self.confidence = load_model_and_configuration(self.experiment)

            if torch.cuda.is_available():
                self.long = torch.cuda.LongTensor
                self.float = torch.cuda.FloatTensor
                device = torch.device('cuda')
            else:
                self.long = torch.LongTensor
                self.float = torch.FloatTensor
                device = torch.device('cpu')

            self.model.to(device)
            self.model.eval()

            self.save_as = "_".join([self.experiment, self.label_type,
                                     self.doc_repr, which_labels, str(threshold)])

        #word embeddings without additional classifiers
        else:
            self.doc_repr = "avg"
            self.label_type = label_type
            self.save_as = "_".join([self.experiment, self.label_type,
                                     self.doc_repr, which_labels, str(threshold)])

        #for IR
        print(str(datetime.datetime.now()).split('.')[0], "Loading collection and query examples")

        self.collection = IR_Dataset("train", label_type, threshold, which_labels)
        self.queries = IR_Dataset("test", label_type, threshold, which_labels)

        print(str(datetime.datetime.now()).split('.')[0], "Evaluating %s... collection %s examples, queries %s examples"
              %(self.save_as, len(self.collection.df), len(self.queries.df)))

        self.run_evaluation()


    def run_classifier_evaluation(self):

        if "mucl" in self.experiment:
            test_dataset = MulticlassDataset(file="test", label_type=self.label_type,
                                                  threshold=self.threshold)
            test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
            test_loss, test_accuracy, true_ids, predicted_ids = test_multiclass_model(
                model=self.model, test_loader=test_loader, num_examples=len(test_dataset.df), long=self.long)

        elif "pair_cel" in self.experiment:
            test_dataset = BinaryPairDataset(file="test", label_type=self.label_type,
                                            threshold=self.threshold)
            test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
            test_loss, test_accuracy = test_pair_cel_model(
                model=self.model, test_loader=test_loader, num_examples=len(test_dataset.df), long=self.long,
                float=self.float)

        elif "triplet" in self.experiment:
            test_dataset = BinaryTripletDataset(file="test", label_type=self.label_type,
                                                threshold=self.threshold)
            test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
            test_loss, test_accuracy = test_triplet_model(
                model=self.model, test_loader=test_loader, num_examples=len(test_dataset.df), long=self.long,
                float=self.float, confidence=self.confidence)

        return test_loss, test_accuracy


    def get_collection_and_queries(self):

        if self.experiment == "TF-IDF":
                collection, queries = compute_tfidf(collection=self.collection.df, queries=self.queries.df, idf_base=self.which_labels)

        elif self.experiment == "BM25":
                queries = list(self.queries.df["TEXT"].apply(lambda x: tokenize_and_clean(x)))
                collection = list(self.collection.df["TEXT"].apply(lambda x: tokenize_and_clean(x)))

        elif self.experiment == "RANDOM_DOC":
            queries = normalize(np.random.uniform(low=-0.1, high=0.1, size=(len(self.queries), 300)).astype("float32"), axis=1)
            collection = normalize(np.random.uniform(low=-0.1, high=0.1, size=(len(self.collection), 300)).astype("float32"), axis=1)

        else:

            collection_loader = DataLoader(self.collection, batch_size=64, shuffle=False)
            queries_loader = DataLoader(self.queries, batch_size=64, shuffle=False)

            # without additional classifier
            if "mucl" not in self.experiment and "pair" not in self.experiment and "triplet" not in self.experiment:

                queries = combine_avg_doc_embs(sequences=queries_loader, model=self.experiment,
                                               idf_base="all")
                collection = combine_avg_doc_embs(sequences=collection_loader, model=self.experiment,
                                              idf_base="all")

            else: # with classifier

                if self.doc_repr == "avg":
                    queries = combine_avg_doc_embs(sequences=queries_loader, model=self.model,
                                                   idf_base=self.which_labels)
                    collection = combine_avg_doc_embs(sequences=collection_loader, model=self.model,
                                                  idf_base=self.which_labels)

                elif self.doc_repr in ["hidden", "linear", "softmax"]:
                    collection = extract_from_model(doc_repr=self.doc_repr, sequences=collection_loader,
                                                model=self.model)
                    queries = extract_from_model(doc_repr=self.doc_repr, sequences=queries_loader,
                                                model=self.model)

        return collection, queries

    def find_top_doc_ids(self):

        collection, queries = self.get_collection_and_queries()

        if self.experiment == "BM25":
            top_doc_ids = bm25(k=len(self.collection.df), queries=queries, collection=collection, idf_base=self.which_labels)

        elif self.doc_repr == "softmax":
            top_doc_ids = jsd(k=None, queries=queries, collection=collection)

        else:
            index = build_index(collection=collection)
            top_doc_ids, _ = search_index(index=index, k=len(self.collection.df), queries=queries)

        return top_doc_ids, collection, queries

    def run_ir_evaluation(self):

        top_doc_ids, collection, queries = self.find_top_doc_ids()

        mean_avg_prec = compute_mean_avg_prec(top_doc_ids=top_doc_ids, train_df=self.collection.df,
                                              test_df=self.queries.df, label_type=self.label_type)

        return mean_avg_prec

    def run_evaluation(self):

        map = self.run_ir_evaluation()
        results = {"MAP": map}
        if self.test_acc_loss:
            test_loss, test_accuracy = self.run_classifier_evaluation()
            results["LOSS"] = test_loss
            results["ACCURACY"] = test_accuracy
        results = pd.DataFrame(results, index=[0]).T
        print(results)
        with open("public_data/results/IR_evaluation_%s.pkl" % self.save_as, 'wb') as out:
            results.to_pickle(out, compression=None, protocol=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--experiment', type=str, help='which experiment/trained model to use, e.g. "5", "8", "TF-IDF", "BM25"...')
    parser.add_argument(
        '--label_type', default="PRODUCT", nargs='?', type=str, help='"PRODUCT", "ISSUE"')
    parser.add_argument(
        '--doc_repr', default="hidden", nargs='?', type=str, help='"avg", "hidden", "linear", "softmax"')
    parser.add_argument(
        '--which_labels', default="all", nargs='?', type=str, help='"above_threshold", "below_threshold", "all"')
    parser.add_argument(
        '--threshold', default=8000, nargs='?', type=int, help='minimum frequency of label in training set')
    parser.add_argument(
        '--test_acc_loss', default=False, nargs="?", type=bool, help='whether to run classifier evaluation')
    args = parser.parse_args()

    print(', '.join('%s: %s' % (k, str(v)) for k, v in dict(vars(args)).items()))

    evaluate = Evaluation_IR(experiment=args.experiment, label_type=args.label_type, doc_repr=args.doc_repr,
                             which_labels=args.which_labels, threshold=args.threshold,
                             test_acc_loss=args.test_acc_loss)

