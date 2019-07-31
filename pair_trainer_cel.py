import pickle
import torch.nn.utils.rnn
from torch.utils.data import DataLoader
import logging
import argparse
import uuid
from classifier_helpers import BinaryPairDataset, IR_Dataset, plot_training, plot_ir, log
from encoders import Encoder
from get_doc_embs import extract_from_model
from sim_search import search_index, build_index
from metrics import compute_mean_avg_prec

class Trainer:
    def __init__(self, experiment, label_type, lr, weight_decay, num_directions, batch_size, num_epochs, threshold, hidden_size,
                 dropout):

        self.experiment = experiment
        self.label_type = label_type
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_directions = num_directions
        self.batch_size = batch_size
        self.num_epochs = num_epochs

        with open("public_data/vocab/word2id.pkl", 'rb') as infile:
            self.word2id = pickle.load(infile)

        self.train_dataset = BinaryPairDataset("train", label_type, threshold)
        self.valid_dataset = BinaryPairDataset("valid", label_type, threshold)

        self.orig_labels = self.train_dataset.orig_labels

        self.collection = IR_Dataset("train", label_type, threshold, "above_threshold")
        self.queries = IR_Dataset("valid", label_type, threshold, "above_threshold")

        with open("public_data/vocab/word2id.pkl", 'rb') as infile:
            self.word2id = pickle.load(infile)

        with open("public_data/embeddings/word_embeddings_%s.pkl" %self.experiment, 'rb') as infile:
            embeddings = pickle.load(infile)

        self.model = Encoder(embeddings=embeddings, dropout=dropout,
                             hidden_size=hidden_size, num_directions=num_directions)

        if torch.cuda.is_available():
            self.model = self.model.cuda()
            self.float = torch.cuda.FloatTensor
            self.long = torch.cuda.LongTensor
        else:
            self.float = torch.FloatTensor
            self.long = torch.LongTensor

        log('============ EXPERIMENT ID ============')
        id = str(uuid.uuid4())[:8]
        log("Files will be stored with id %s" % id)

        self.save_as = "_".join([self.experiment, label_type, str(threshold), "pair_cel", id])

        with open("public_data/models/configuration_classifier_%s.pkl" % self.save_as, 'wb') as out:
            pickle.dump({"NUM_TRAIN": len(self.train_dataset.df), "NUM_VALID": len(self.valid_dataset.df),
                         "DROPOUT": dropout, "HIDDEN_SIZE": hidden_size, "NUM_DIR": num_directions,
                         "EMB_SHAPE": embeddings.shape, "LABEL_TYPE": label_type,
                         "THRESHOLD": threshold, "LABELS": self.orig_labels}, out)

        self.run_training()

    def train(self, train_loader):
        """
        Trains the model
        :param train_loader: training documents
        :return: training acc & loss
        """

        self.model.train()
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr,
                                    weight_decay=self.weight_decay)
        loss_func = torch.nn.CosineEmbeddingLoss()
        sum_loss = 0

        for batch_id, (text_a, text_b, label) in enumerate(train_loader):
            last_hidden_a = self.model(text_a.type(self.long))
            last_hidden_b = self.model(text_b.type(self.long))

            for i in range(len(label)):
                if label[i] == 0.0:
                    label[i] = -1.0

            cur_loss = loss_func(last_hidden_a, last_hidden_b, label.type(self.float))
            sum_loss += cur_loss.data
            cur_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        train_loss = sum_loss/len(self.train_dataset.df)

        return train_loss

    def validate(self, valid_loader):
        """
        Validates model
        :param valid_loader: validation documents
        :return: validation acc & loss
        """
        self.model.eval()
        loss_func = torch.nn.CosineEmbeddingLoss()
        sum_loss = 0

        for batch_id, (text_a, text_b, label) in enumerate(valid_loader):
            last_hidden_a = self.model(text_a.type(self.long))
            last_hidden_b = self.model(text_b.type(self.long))

            for i in range(len(label)):
                if label[i] == 0.0:
                    label[i] = -1.0

            cur_loss = loss_func(last_hidden_a, last_hidden_b, label.type(self.float))
            sum_loss += cur_loss.data

        valid_loss = sum_loss / len(self.valid_dataset.df)

        return valid_loss

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

        map = compute_mean_avg_prec(top_doc_ids=top_doc_ids, train_df=self.collection.df,
                                    test_df=self.queries.df, label_type=self.label_type)

        return map

    def run_training(self):
        """
        Runs the training and validation, computes the MAP, plots the results
        """

        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        valid_loader = DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=True)

        collection_loader = DataLoader(self.collection, batch_size=64, shuffle=False)
        queries_loader = DataLoader(self.queries, batch_size=64, shuffle=False)

        log('============ IR DATA ============')
        log("Corpus examples: %d, query examples: %d" %(len(self.collection.df), len(self.queries.df)))

        log('============ TRAINING & VALIDATION ============')
        log("%d original classes, %d training examples, %d validation examples" % (
        len(self.orig_labels), len(self.train_dataset.df), len(self.valid_dataset.df)))

        train_losses = []
        valid_losses = []
        mean_avg_precs = []

        best_map = -1

        for epoch in range(self.num_epochs):

            log("Epoch: %d/%d ..." % (epoch, self.num_epochs))

            train_loss = self.train(train_loader)
            valid_loss = self.validate(valid_loader)

            train_losses.append(train_loss)
            valid_losses.append(valid_loss)

            map = self.validate_ir(collection_loader, queries_loader)
            mean_avg_precs.append(map)

            if map > best_map:
                best_map = map
                torch.save(self.model.state_dict(), "public_data/models/experiment_%s/classifier_%s.pkl"
                           % (self.save_as.split("_")[0], self.save_as))

            log("TrainLoss: %.3f " %train_loss + "ValidLoss: %.3f " %valid_loss + "MAP: %.3f " %map )

            plot_training(train_losses, valid_losses, "loss", "_".join([self.save_as, "loss"]))
            plot_ir(values=mean_avg_precs, save_as=self.save_as, title="Mean Average Precision", name="map")

        return train_losses, valid_losses


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--experiment', type=str, help='which embedding model is used, 1-2 or 5-10_svd/gan')
    parser.add_argument(
        '--threshold', type=int, nargs="?", default=8000, help='minimum frequency of label in training set to use as class')
    parser.add_argument(
        '--batch_size', type=int, nargs="?", default=16, help='batchsize')
    parser.add_argument(
        '--label_type', type=str, nargs="?", default="PRODUCT", help='"PRODUCT", "ISSUE", "COMB"')
    parser.add_argument(
        '--hidden_size', type=int, nargs="?", default=300, help='dimension of hidden layers')
    parser.add_argument(
        '--num_directions', type=int, nargs="?", default=2, help='bi/unidirectional lstm')
    parser.add_argument(
        '--dropout', type=float, nargs="?", default=0.0, help='dropout probability')
    parser.add_argument(
        '--lr', type=float, nargs="?", default=0.01, help='learning rate')
    parser.add_argument(
        '--weight_decay', type=float, nargs="?", default=0.00001, help='learning rate')
    parser.add_argument(
        '--num_epochs', type=int, nargs="?", default=75, help='total number of training epochs')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, filename="public_data/logs_and_plots/log_%s_pair_cel.log" % args.experiment,
                        filemode="a+", format="%(asctime)-15s %(levelname)-8s %(message)s")

    log('============ PARAMETERS ============')
    log(', '.join('%s: %s' % (k, str(v)) for k, v in dict(vars(args)).items()))

    train = Trainer(experiment=args.experiment, label_type=args.label_type,
                    threshold=args.threshold, hidden_size=args.hidden_size,
                    num_directions= args.num_directions, weight_decay=args.weight_decay,
                    batch_size=args.batch_size, dropout=args.dropout,
                    lr=args.lr, num_epochs=args.num_epochs)


