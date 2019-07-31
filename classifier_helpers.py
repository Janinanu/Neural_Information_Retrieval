import pickle
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from custom_tokenizer import tokenize_and_clean
from multiclass_classifier import Classifier
from encoders import Encoder, TripletEncoder
np.seterr(divide='ignore', invalid='ignore')

CORPUS_LIMIT = 1500
QUERY_LIMIT = 150
TRAIN_LIMIT = 15000
VALID_LIMIT = 1500

class IR_Dataset(Dataset):
    """
    Loads collection or queries from file, optionally applies threshold 
    """

    def __init__(self, file, label_type, threshold, which_labels):
        self.label_type = label_type

        with open('public_data/vocab/word2id.pkl', 'rb') as infile:
            self.word2id = pickle.load(infile)

        with open('public_data/inputs/%s.pkl' %file, 'rb') as infile:
            data = pd.read_pickle(infile, compression=None)
        if file == "train":
            self.df = data[:TRAIN_LIMIT]
        elif file in ["valid", "test"]:
            self.df = data[:VALID_LIMIT]

        with open('public_data/stats/stats_train.pkl', 'rb') as stats:
            stats = pickle.load(stats)

        if which_labels in ["above_threshold", "below_threshold"]:

            if which_labels == "above_threshold":
                labels = [label for label, freq in
                          stats["DISTR_" + label_type].items() if freq >= threshold]

            elif which_labels == "below_threshold":
                labels = [label for label, freq in
                          stats["DISTR_" + label_type].items() if freq < threshold]

            self.df = self.df[self.df[label_type].isin(labels)]

        if file == "train":
            self.df = self.df[:CORPUS_LIMIT]
        elif file in ["valid", "test"]:
            self.df = self.df[:QUERY_LIMIT]

        self.sequences = [torch.LongTensor([self.word2id.get(word, 1) for word in tokenize_and_clean(text)])
                        for text in self.df["TEXT"]]
        self.sequences = pad_sequence(self.sequences, batch_first=True)

    def __len__(self):

        return len(self.df)

    def __getitem__(self, index):

        text = self.sequences[index, :]

        return text


class BinaryTripletDataset(Dataset):
    """Loads document triplets from training/validation/test file, applies threshold
    """

    def __init__(self, file, label_type, threshold):

        with open('public_data/vocab/word2id.pkl', 'rb') as infile:
            self.word2id = pickle.load(infile)

        with open('public_data/stats/stats_train.pkl', 'rb') as stats:
            stats = pickle.load(stats)
        self.orig_labels = [label for label, freq in stats["DISTR_" + label_type].items() if freq >= threshold]
        self.label_type = label_type

        with open("public_data/inputs/binary_%s_%s_%s_%s.pkl" % (file, label_type, str(threshold), "triplet"),'rb') as indata:
            self.df = pd.read_pickle(indata, compression=None)[:TRAIN_LIMIT]

        self.sequences_A = [torch.LongTensor([self.word2id.get(word, 1) for word in tokenize_and_clean(text)])
                     for text in self.df["TEXT_A"]]
        self.sequences_A = pad_sequence(self.sequences_A, batch_first=True)

        self.sequences_B = [torch.LongTensor([self.word2id.get(word, 1) for word in tokenize_and_clean(text)])
                       for text in self.df["TEXT_B"]]
        self.sequences_B = pad_sequence(self.sequences_B, batch_first=True)

        self.sequences_C = [torch.LongTensor([self.word2id.get(word, 1) for word in tokenize_and_clean(text)])
                       for text in self.df["TEXT_C"]]
        self.sequences_C = pad_sequence(self.sequences_C, batch_first=True)

        self.targets = torch.FloatTensor(np.ones((len(self.df))))

    def __len__(self):

        return len(self.df)

    def __getitem__(self, index):

        text_A = self.sequences_A[index, :]
        text_B = self.sequences_B[index, :]
        text_C = self.sequences_C[index, :]
        targets = self.targets[index]

        return text_A, text_B, text_C, targets


class BinaryPairDataset(Dataset):
    """Loads document pairs from training/validation/test file, applies threshold
    """

    def __init__(self, file, label_type, threshold):

        with open('public_data/vocab/word2id.pkl', 'rb') as infile:
            self.word2id = pickle.load(infile)

        with open('public_data/stats/stats_train.pkl', 'rb') as stats:
            stats = pickle.load(stats)
        self.orig_labels = [label for label, freq in stats["DISTR_" + label_type].items() if freq >= threshold]
        self.label_type = label_type

        with open("public_data/inputs/binary_%s_%s_%s_%s.pkl" % (file, label_type, str(threshold), "pair"),'rb') \
                as indata:
            self.df = pd.read_pickle(indata, compression=None)[:TRAIN_LIMIT]

        self.sequences_A = [torch.LongTensor([self.word2id.get(word, 1) for word in tokenize_and_clean(text)])
                            for text in self.df["TEXT_A"]]
        self.sequences_A = pad_sequence(self.sequences_A, batch_first=True)

        self.sequences_B = [torch.LongTensor([self.word2id.get(word, 1) for word in tokenize_and_clean(text)])
                            for text in self.df["TEXT_B"]]
        self.sequences_B = pad_sequence(self.sequences_B, batch_first=True)

        self.labels = torch.FloatTensor(list(self.df["BINARY_LABEL"]))

    def __len__(self):

        return len(self.df)

    def __getitem__(self, index):

        text_A = self.sequences_A[index, :]
        text_B = self.sequences_B[index, :]
        label = self.labels[index]

        return text_A, text_B, label


class MulticlassDataset(Dataset):
    """Loads multiclass documents from training/validation/test file, applies threshold
    """

    def __init__(self, file, label_type, threshold):

        with open('public_data/vocab/word2id.pkl', 'rb') as infile:
            self.word2id = pickle.load(infile)

        with open('public_data/stats/stats_train.pkl', 'rb') as stats:
            stats = pickle.load(stats)
        d = stats["DISTR_" + label_type]
        labels = [label for label, freq in sorted(d.items(), key=lambda item: item[1], reverse=True)
                  if freq >= threshold]
        self.label2id = {l: i for i, l in enumerate(labels)}

        with open('public_data/inputs/%s.pkl' %file, 'rb') as indata:
            if file == "train":
                data = pd.read_pickle(indata, compression=None)[:TRAIN_LIMIT]
            elif file in ["valid", "test"]:
                data = pd.read_pickle(indata, compression=None)[:VALID_LIMIT]
        self.df = data[data[label_type].isin(labels)]

        self.sequences = [torch.LongTensor([self.word2id.get(word, 1) for word in tokenize_and_clean(text)])
                          for text in self.df["TEXT"]]
        self.sequences = pad_sequence(self.sequences, batch_first=True)

        self.labels = torch.LongTensor([self.label2id[label] for label in self.df[label_type]])

    def __len__(self):

        return len(self.df)

    def __getitem__(self, index):

        text = self.sequences[index, :]
        label = self.labels[index]

        return text, label

def log(to_log):
    logging.info(to_log)
    print(to_log)


def plot_training(train_values, valid_values, loss_or_acc, save_as):
    """
    Plots accuracy and loss during training and validation and stores the plots
    :param train_values: list of train acc/ values for each epoch
    :param valid_values: list of validation acc/ values for each epoch
    :param loss_or_acc: "loss" or "acc"
    :param save_as: contains experiment ID
    """
    plt.plot(train_values, "#34495e", label="Train")
    plt.plot(valid_values, "#2ecc71", label="Valid")
    plt.xlabel("Epoch")
    plt.legend(loc='upper left')
    plt.title(loss_or_acc)
    plt.savefig("public_data/logs_and_plots/%s.png" %save_as)
    plt.show()
    plt.close()


def plot_ir(values, title, save_as, name):
    """
     Plots mean average precision during training and stores the plot
     :param values: list of MAP for each epoch
     :param title: set title
     :param save_as: contains experiment ID
     :param name: "map"
     """
    plt.plot(values, "#34495e")
    plt.xlabel("Epoch")
    plt.legend(loc='upper left')
    plt.title(title)
    plt.savefig("public_data/logs_and_plots/%s_%s.png" %(save_as, name))
    plt.show()
    plt.close()


def load_model_and_configuration(model_name):
    """
    Loads configuration and skeleton of trained model
    :param model_name: string ID of trained model
    :return: loaded model and configuration values
    """

    with open("public_data/models/configuration_classifier_%s.pkl" % model_name, 'rb') as infile:
        configuration = pickle.load(infile)

    if "mucl" in model_name:
        model = Classifier(embeddings=np.zeros(configuration["EMB_SHAPE"]),
                           num_classes=len(configuration["LABEL2ID"]),
                           dropout=configuration["DROPOUT"],
                           hidden_size=configuration["HIDDEN_SIZE"],
                           num_directions=configuration["NUM_DIR"])
        load_model(model_name, model)
        label2id = configuration["LABEL2ID"]
        labels = configuration["LABELS"]
        label_type = configuration["LABEL_TYPE"]
        return model, label_type, label2id, labels

    elif "pair_cel" in model_name:
        model = Encoder(embeddings=np.zeros(configuration["EMB_SHAPE"]),
                          dropout=configuration["DROPOUT"],
                          hidden_size=configuration["HIDDEN_SIZE"],
                          num_directions=configuration["NUM_DIR"])
        load_model(model_name, model)
        label_type = configuration["LABEL_TYPE"]
        threshold = configuration["THRESHOLD"]
        return model, label_type, threshold

    elif "triplet" in model_name:
        encoder = Encoder(embeddings=np.zeros(configuration["EMB_SHAPE"]),
                          dropout=configuration["DROPOUT"],
                          hidden_size=configuration["HIDDEN_SIZE"],
                          num_directions=configuration["NUM_DIR"])
        model = TripletEncoder(encoder)
        load_model(model_name, model)
        vfc_type = configuration["LABEL_TYPE"]
        threshold = configuration["THRESHOLD"]
        confidence = configuration["CONFIDENCE"]
        return model, vfc_type, threshold, confidence

def load_model(model_name, model):
    """
    Loads the state dict onto the model skeleton
    :param model_name: string ID of model
    :param model: preloaded skeleton of model
    """
    emb_nr = model_name.split("_")[0]  # 8
    model_path = "public_data/models/experiment_%s/classifier_%s.pkl" % (emb_nr, model_name)
    model.load_state_dict(torch.load(model_path))

