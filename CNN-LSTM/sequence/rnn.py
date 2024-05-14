"""
Simple models for encoding dense representations of sequences
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


################################################################################
# Recurrent Neural Network Models
################################################################################

class RNN(nn.Module):

    def __init__(self, n_classes, input_size, hidden_size, rnn_type="LSTM", dropout=0.0,
                 max_seq_len=15, attention=True, bidirectional=True, use_cuda=False):
        """
        Initalize RNN module

        :param n_classes:
        :param input_size:
        :param hidden_size:
        :param rnn_type:    GRU or LSTM
        :param dropout:
        :param max_seq_len:
        :param attention:
        :param bidirectional:
        :param use_cuda:
        """
        super(RNN, self).__init__()

        self.input_size    = input_size
        self.hidden_size   = hidden_size
        self.bidirectional = bidirectional
        self.n_classes     = n_classes
        self.attention     = attention
        self.max_seq_len   = max_seq_len
        self.use_cuda      = use_cuda

        self.rnn_type = rnn_type
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(input_size, hidden_size, batch_first=True,
                                             dropout=dropout, bidirectional=self.bidirectional)
        self.dropout = nn.Dropout(dropout)

        b = 2 if self.bidirectional else 1

        if attention:
            self.attn_linear_w_1 = nn.Linear(b * hidden_size, b * hidden_size, bias=True)
            self.attn_linear_w_1a = nn.Linear(b * hidden_size, b * hidden_size, bias=True)
            self.attn_linear_w_2 = nn.Linear(b * hidden_size, 1, bias=False)
            self.attn_linear_w_2a = nn.Linear(b * hidden_size, 1, bias=False)
            self.alpha = nn.Parameter(torch.zeros(1), requires_grad=True)

        self.linear = nn.Linear(b * hidden_size, n_classes)

    def embedding(self, x, hidden, x_mask=None):
        """
        Get learned representation
        """
        x_mask = self._get_mask(x) if not x_mask else x_mask

        output, hidden = self.rnn(x, hidden)
        output = self.dropout(output)

        if self.attention:
            output = self._two_fold_attn_pooling(output)
        else:
            output = self._mean_pooling(output, x_mask)

        return output

    def _mean_pooling(self, x, x_mask):
        """
        Mean pooling of RNN hidden states
        """
        return torch.mean(x, 1)
        # x_lens = x_mask.data.eq(0).long().sum(dim=1)
        # if self.use_cuda:
        #     weights = Variable(torch.ones(x.size()).cuda() / x_lens.unsqueeze(1).float())
        # else:
        #     weights = Variable(torch.ones(x.size()) / x_lens.unsqueeze(1).float())
        # weights.data.masked_fill_(x_mask.data, 0.0)
        # output = torch.bmm(x.transpose(1, 2), weights.unsqueeze(2)).squeeze(2)
        # return output

    def _attn_mean_pooling(self, x, x_mask):
        """
        Weighted mean pooling of RNN hidden states, where weights are
        calculated via an attention layer where the attention weight is
            a = T' . tanh(Wx + b)
            where x is the input, b is the bias.
        """
        emb_squish = F.tanh(self.attn_linear_w_1(x))
        emb_attn = self.attn_linear_w_2(emb_squish)
        emb_attn.data.masked_fill_(x_mask.unsqueeze(2).data, float("-inf"))
        emb_attn_norm = F.softmax(emb_attn.squeeze(2), dim=0)
        emb_attn_vectors = torch.bmm(x.transpose(1, 2), emb_attn_norm.unsqueeze(2)).squeeze(2)
        return emb_attn_vectors
        
    def _two_fold_attn_pooling(self, x):
        emb_squish_1 = F.tanh(self.attn_linear_w_1(x))
        emb_attn_1 = self.attn_linear_w_2(emb_squish_1)
        emb_squish_2 = F.tanh(self.attn_linear_w_1a(x))
        emb_attn_2 = self.attn_linear_w_2a(emb_squish_2)
        alpha_limit = F.sigmoid(self.alpha)
        emb_attn = alpha_limit*emb_attn_1 + (1-alpha_limit)*emb_attn_2
        emb_attn_norm = F.softmax(emb_attn.squeeze(2), dim=0)
        emb_attn_vectors = torch.bmm(x.transpose(1, 2), emb_attn_norm.unsqueeze(2)).squeeze(2)
        return emb_attn_vectors

    def _get_mask(self, x):
        """
        Return an empty mask
        :param x:
        :return:
        """
        x_mask = Variable(torch.zeros(x.size(0), self.max_seq_len).byte())
        return x_mask.cuda() if self.use_cuda else x_mask

    def forward(self, x, hidden, x_mask=None):
        """
        Forward pass of the network

        :param x:
        :param hidden:
        :param x_mask: 0-1 byte mask for variable length sequences
        :return:
        """
        x_mask = self._get_mask(x) if not x_mask else x_mask

        output, hidden = self.rnn(x, hidden)
        output = self.dropout(output)

        if self.attention:
            output = self._two_fold_attn_pooling(output)
        else:
            output = self._mean_pooling(output, x_mask)

        output = self.linear(output)
        return output

    def init_hidden(self, batch_size):
        """
        Initialize hidden state params

        :param batch_size:
        :return:
        """
        b = 2 if self.bidirectional else 1
        if self.rnn_type  == "LSTM":
            h0 = (Variable(torch.zeros(b, batch_size, self.hidden_size)),
                  Variable(torch.zeros(b, batch_size, self.hidden_size)))
            h0 = h0 if not self.use_cuda else [h0[0].cuda(), h0[1].cuda()]
        else:
            h0 = Variable(torch.zeros(b, batch_size, self.hidden_size))
            h0 = h0 if not self.use_cuda else h0.cuda()
        return h0


class MetaRNN(RNN):
    """
    RNN class for Meta data concatenating into seq_output before classifier
    """
    def forward(self, x, hidden, x_mask=None):
        """
        Forward pass of the network

        :param x:
        :param hidden:
        :param x_mask: 0-1 byte mask for variable length sequences
        :return:
        """
        x_mask = self._get_mask(x) if not x_mask else x_mask

        output, hidden = self.rnn(x, hidden)
        output = self.dropout(output)

        if self.attention:
            output = self._attn_mean_pooling(output, x_mask)
        else:
            output = self._mean_pooling(output, x_mask)

        return output
