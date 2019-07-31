import pickle
import torch.nn.utils.rnn
from torch.utils.data import DataLoader
import logging
import argparse
import uuid
from classifier_helpers import IR_Dataset, log, MulticlassDataset, plot_training, plot_ir
from multiclass_classifier import Classifier
from get_doc_embs import extract_from_model
from sim_search import search_index, build_index
from metrics import compute_mean_avg_prec


class Trainer:
    def __init__(self, experiment, label_type, lr, weight_decay, num_directions, batch_size,
                 num_epochs, threshold, hidden_size, dropout):

        self.experiment = experiment
        self.label_type = label_type
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.num_epochs = num_epochs

        self.train_dataset = MulticlassDataset(file="train", label_type=label_type, threshold=threshold)
        self.valid_dataset = MulticlassDataset(file="valid", label_type=label_type, threshold=threshold)
        self.label2id = self.train_dataset.label2id

        #for validating MAP during classifier training
        self.collection = IR_Dataset(file="train", label_type=label_type, threshold=threshold, which_labels="above_threshold")
        self.queries = IR_Dataset(file="valid", label_type=label_type, threshold=threshold, which_labels="above_threshold")

        with open("public_data/vocab/word2id.pkl", 'rb') as infile:
            self.word2id = pickle.load(infile)

        with open("public_data/embeddings/word_embeddings_%s.pkl" %self.experiment, 'rb') as infile:
            embeddings = pickle.load(infile)

        if torch.cuda.is_available():
            self.long = torch.cuda.LongTensor
            self.model = Classifier(embeddings=embeddings, num_classes=len(self.label2id),
                                    dropout=dropout, hidden_size=hidden_size,
                                    num_directions=num_directions).cuda()
        else:
            self.long = torch.LongTensor
            self.model = Classifier(embeddings=embeddings, num_classes=len(self.label2id),
                                    dropout=dropout, hidden_size=hidden_size,
                                    num_directions=num_directions)

        log('============ EXPERIMENT ID ============')
        id = str(uuid.uuid4())[:8]
        log("Files will be stored with id %s" % id)
        self.save_as = "_".join([self.experiment, "mucl", label_type, str(threshold), id])

        with open("public_data/models/configuration_classifier_%s.pkl" % self.save_as, 'wb') as out:
            pickle.dump({"NUM_TRAIN": len(self.train_dataset.df), "NUM_VALID": len(self.valid_dataset.df),
                         "DROPOUT": dropout, "HIDDEN_SIZE": hidden_size, "EMB_SHAPE": embeddings.shape,
                         "LABEL_TYPE": label_type, "THRESHOLD": threshold, "LABELS": list(self.label2id.keys()),
                         "LABEL2ID": self.label2id, "NUM_CLASSES": len(self.label2id), "NUM_DIR": num_directions}, out)

        self.run_training()

    def train(self, train_loader):
        """
        Trains the model
        :param train_loader: training documents
        :return: training acc & loss
        """

        self.model.train()
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr,
                                    weight_decay=self.weight_decay)
        loss_func = torch.nn.CrossEntropyLoss()
        sum_loss = 0
        sum_correct = 0

        for batch_id, (texts, labels) in enumerate(train_loader):

            _, hidden_scores = self.model(texts.type(self.long))
            labels = labels.type(self.long)
            cur_loss = loss_func(hidden_scores, labels)
            sum_loss += cur_loss.data

            cur_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            _, prediction = hidden_scores.max(dim=-1)
            sum_correct += sum([1 for i in range(len(labels)) if labels[i] == prediction[i]])

        train_loss = sum_loss/len(self.train_dataset.df)
        train_accuracy = sum_correct/len(self.train_dataset.df)

        return train_loss, train_accuracy

    def validate(self, valid_loader):
        """
        Validates model
        :param valid_loader: validation documents
        :return: validation acc & loss
        """
        self.model.eval()
        loss_func = torch.nn.CrossEntropyLoss()
        sum_loss = 0
        sum_correct = 0

        for batch_id, (texts, labels) in enumerate(valid_loader):

            _, hidden_scores = self.model(texts.type(self.long))
            labels = labels.type(self.long)

            cur_loss = loss_func(hidden_scores, labels)
            sum_loss += cur_loss.data

            _, prediction = hidden_scores.max(dim=-1)
            sum_correct += sum([1 for i in range(len(labels)) if labels[i] == prediction[i]])

        valid_loss = sum_loss/len(self.valid_dataset.df)
        valid_accuracy = sum_correct/len(self.valid_dataset.df)

        return valid_loss, valid_accuracy

    def validate_ir(self, collection_loader, queries_loader):
        """
        Computes MAP on last hidden states during classifier training
        :param collection_loader: documents in the collection
        :param queries_loader: query documents
        :return: mean average precision
        """
        collection = extract_from_model(doc_repr="hidden", sequences=collection_loader, model=self.model)
        queries = extract_from_model(doc_repr="hidden", sequences=queries_loader, model=self.model)

        index = build_index(collection=collection)
        top_doc_ids, _ = search_index(index=index, k=len(self.collection.df), queries=queries)

        mean_avg_prec = compute_mean_avg_prec(top_doc_ids=top_doc_ids, train_df=self.collection.df,
                                              test_df=self.queries.df, label_type=self.label_type)

        return mean_avg_prec


    def run_training(self):
        """
        Runs the training and validation, computes the MAP, plots the results
        """
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        valid_loader = DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=True)

        collection_loader = DataLoader(self.collection, batch_size=64, shuffle=False)
        queries_loader = DataLoader(self.queries, batch_size=64, shuffle=False)

        log('============ IR DATA ============')
        log("Corpus examples: %d, query examples: %d" % (len(self.collection.df), len(self.queries.df)))

        log('============ TRAINING & VALIDATION ============')
        log("%d classes, %d training examples, %d validation examples" %(len(self.label2id), len(self.train_dataset.df),
                                                                        len(self.valid_dataset.df)))

        train_losses = []
        valid_losses = []
        train_accs = []
        valid_accs = []
        mean_avg_precs = []

        best_map = -1

        for epoch in range(self.num_epochs):

            log("Epoch: %d/%d ..." % (epoch, self.num_epochs))

            train_loss, train_accuracy = self.train(train_loader)
            valid_loss, valid_accuracy = self.validate(valid_loader)

            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
            train_accs.append(train_accuracy)
            valid_accs.append(valid_accuracy)

            map = self.validate_ir(collection_loader, queries_loader)
            mean_avg_precs.append(map)

            if map > best_map:

                best_map = map
                torch.save(self.model.state_dict(), "public_data/models/experiment_%s/classifier_%s.pkl"
                            %(self.save_as.split("_")[0], self.save_as))

            log("TrainLoss: %.3f " % train_loss + "TrainAccuracy: %.3f " % train_accuracy + \
                "ValidLoss: %.3f " % valid_loss + "ValidAccuracy: %.3f " % valid_accuracy + \
                "MAP: %.3f " % map )

            plot_training(train_losses, valid_losses, "Loss", "_".join([self.save_as, "loss"]))
            plot_training(train_accs, valid_accs, "Accuracy", "_".join([self.save_as, "acc"]))
            plot_ir(values=mean_avg_precs, save_as=self.save_as, title="Mean Average Precision", name="map")

        return train_losses, valid_losses, train_accs, valid_accs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--experiment', type=str, help='which embedding model is used, 1-2 or 5-10_svd/gan')
    parser.add_argument(
        '--threshold', type=int, nargs="?", default=8000, help='minimum frequency of label in training set to use as class')
    parser.add_argument(
        '--label_type', type=str, nargs="?", default="PRODUCT", help='"PRODUCT", "ISSUE", "COMB"')
    parser.add_argument(
        '--num_directions', type=int, nargs="?", default=2, help='uni/bidir. LSTM')
    parser.add_argument(
        '--hidden_size', type=int, nargs="?", default=300, help='dimension of hidden layers')
    parser.add_argument(
        '--batch_size', type=int, nargs="?", default=128, help='number of examples per batch')
    parser.add_argument(
        '--dropout', type=float, nargs="?", default=0.0, help='dropout probability')
    parser.add_argument(
        '--lr', type=float, nargs="?", default=0.01, help='learning rate')
    parser.add_argument(
        '--weight_decay', type=float, nargs="?", default=0.00001, help='learning rate')
    parser.add_argument(
        '--num_epochs', type=int, nargs="?", default=75, help='total number of training epochs')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, filename="public_data/logs_and_plots/log_%s_mucl.log" %args.experiment,
                        filemode="a+", format="%(asctime)-15s %(levelname)-8s %(message)s")

    log('============ PARAMETERS ============')
    log(', '.join('%s: %s' % (k, str(v)) for k, v in dict(vars(args)).items()))

    trainer = Trainer(experiment=args.experiment, label_type=args.label_type,
                    threshold=args.threshold, hidden_size=args.hidden_size,
                    num_directions=args.num_directions, weight_decay=args.weight_decay,
                    dropout=args.dropout, lr=args.lr, batch_size=args.batch_size,
                    num_epochs=args.num_epochs)



