import torch.nn as nn
import torch
import torch.nn.utils.rnn

class Classifier(nn.Module):

    def __init__(self, embeddings, num_classes, dropout, hidden_size, num_directions):

        super(Classifier, self).__init__()

        self.num_classes = num_classes
        self.emb_size = embeddings.shape[1]
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.num_directions = num_directions
        if self.num_directions == 2:
            bidirectional = True
        elif self.num_directions == 1:
            bidirectional = False

        self.embeddings = nn.Embedding.from_pretrained(torch.FloatTensor(embeddings))
        self.lstm = nn.LSTM(self.emb_size, self.hidden_size, batch_first=True, bidirectional=bidirectional)
        self.dropout_layer = nn.Dropout(self.dropout)
        self.linear = nn.Linear(num_directions*self.hidden_size, self.num_classes)
        self.embeddings.weight.requires_grad = True

    def forward(self, inputs): #batch x seq_len
        embeddings = self.embeddings(inputs)  # batch x seq_len x emb_size
        _, (last_hidden, _) = self.lstm(self.dropout_layer(embeddings))  # num_dir x batch x hidden_size
        if self.num_directions == 2:
            last_hidden = torch.cat((last_hidden[0], last_hidden[-1]), 1)  # batch x 2*hidden_size
        elif self.num_directions == 1:
            last_hidden = last_hidden.squeeze(0)  # batch x hidden_size
        linear_scores = self.linear(self.dropout_layer(last_hidden))  # batch x num_classes

        return last_hidden, linear_scores #batch x numdir*hiddensize #batch x num_classes


