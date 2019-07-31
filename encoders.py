import torch.nn as nn
import torch
import torch.nn.utils.rnn



class Encoder(nn.Module):
    """
    Encodes an input sequence with a LSTM and returns the last hidden state,
    used to encode the document pairs and triplets
    """

    def __init__(self, embeddings, dropout, hidden_size, num_directions):
        super(Encoder, self).__init__()

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

        self.embeddings.weight.requires_grad = True

    def forward(self, inputs): #batch x seq_len
        embeddings = self.embeddings(inputs)  # batch x seq_len x emb_size
        _, (last_hidden, _) = self.lstm(self.dropout_layer(embeddings))  # numdir x batch x hiddensize

        if self.num_directions == 2:
            last_hidden = torch.cat((last_hidden[0], last_hidden[-1]), 1)  # batch x 2*hidden_size
        elif self.num_directions == 1:
            last_hidden = last_hidden.squeeze(0)  # batch x hidden_size

        return last_hidden #batch x numdir*hiddensize


class PairEncoder(nn.Module):
    """
    Encodes the document pair a & b and computes their dot product as similarity score
    """

    def __init__(self, encoder):
        super(PairEncoder, self).__init__()
        self.encoder = encoder
        self.hidden_size = self.encoder.hidden_size
        self.num_directions = self.encoder.num_directions

    def forward(self, doc_a, doc_b):
        last_hidden_a = self.encoder(doc_a) # batch_size x numdir*hidden_size
        last_hidden_b = self.encoder(doc_b) # batch_size x numdir*hidden_size

        last_hidden_a = last_hidden_a.view(-1, 1, self.num_directions * self.hidden_size)  # batch_size x 1 x numdir*hidden_size
        last_hidden_b = last_hidden_b.view(-1, self.num_directions * self.hidden_size, 1)  # batch_size x numdir*hidden_size x 1

        score = torch.bmm(last_hidden_a, last_hidden_b)[0, :, :]  # dimensions: (batch_size x 1 x 1) and lastly --> (batch_size)

        return score


class TripletEncoder(nn.Module):
    """Encodes the document triplets a, b, & c and computes the similarity score for a & b and
     a & c using the dot product"""

    def __init__(self, encoder):
        super(TripletEncoder, self).__init__()
        self.encoder = encoder
        self.hidden_size = self.encoder.hidden_size
        self.num_directions = self.encoder.num_directions

    def forward(self, doc_a, doc_b, doc_c):
        last_hidden_a = self.encoder(doc_a)
        last_hidden_b = self.encoder(doc_b)
        last_hidden_c = self.encoder(doc_c)

        last_hidden_a = last_hidden_a.view(-1, 1, self.num_directions * self.hidden_size)  # batch_size x 1 x numdir*hidden_size
        last_hidden_b = last_hidden_b.view(-1, self.num_directions * self.hidden_size, 1)  # batch_size x numdir*hidden_size x 1
        last_hidden_c = last_hidden_c.view(-1, self.num_directions * self.hidden_size, 1)  # batch_size x numdir*hidden_size x 1

        score_1 = torch.bmm(last_hidden_a, last_hidden_b)[:, 0, 0]  # dimensions: (batch_size x 1 x 1) and lastly --> (batch_size)
        score_2 = torch.bmm(last_hidden_a, last_hidden_c)[:, 0, 0]  # dimensions: (batch_size x 1 x 1) and lastly --> (batch_size)

        return score_1, score_2