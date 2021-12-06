#!/usr/bin/env python3

from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
import random

# Base code provided by Marc Canby; modified by Chase Adams 2021
class GRU(nn.Module):
    def __init__(self, feat_size, embed_size, hidden_size, dropout, bidirectional, num_classes):
        super(GRU, self).__init__()

        self.hidden_size = hidden_size
        self.criterion = nn.CrossEntropyLoss()

        # Create an embedding layer (https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html)
        #   to represent the words in your vocabulary. Make sure to use vocab_size, embed_size, and pad_idx here.
        self.linear1 = nn.Linear(feat_size, embed_size)

        # Create a recurrent network (nn.LSTM or nn.GRU) with batch_first = True
        self.gru = nn.GRU(embed_size, self.hidden_size, 2, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        
        # Create a dropout layer (nn.Dropout) using dropout
        self.dropout = nn.Dropout(dropout)

        # Define a linear layer (nn.Linear) that consists of num_classes units 
        #   and takes as input the output of the last timestep. In the bidirectional case, you should concatenate
        #   the output of the last timestep of the forward direction with the output of the last timestep of the backward direction).
        
        self.linear_input_dim = self.hidden_size if not bidirectional else self.hidden_size * 2
        self.linear = nn.Linear(self.linear_input_dim, num_classes)

        self.acc_arr = list()


    def forward(self, x, last_idxes=None): # x: bs x seq length (containg tokens), last_idxes: (seq_length,)

        x = self.linear1(x.float()) # bs x seq length x embed dim

        output, _  = self.gru(x) # bs x seq length x hidden size
        if last_idxes is not None: # training
            idxes = last_idxes.repeat(self.linear_input_dim,1).permute(1,0).unsqueeze(1) # bs x 1 x hidden size
            y = torch.gather(output, 1, idxes.long()) # bs x 1 x hidden size
            assert y.shape == (x.shape[0], 1, self.linear_input_dim)
            y = y.squeeze(1) # bs x hidden size
        else: # inference
            assert x.shape[0] == 1 # assuming inference has batch size of 1
            y = output[:,-1,:]
        assert len(y.shape) == 2 and y.shape[0] == x.shape[0]
        z = self.linear(y) # bs x num_classes
        return z

    def accuracy(self, y_pred, y_true):
        return (torch.argmax(y_pred, dim = 1) == y_true).float().mean()

    def test(self, X_test, y_test):

        y_pred = list()
        for test_ex in X_test:
            test_ex = test_ex.unsqueeze(0) # (1, seq_len)
            output = self.forward(test_ex) # (1, num classes)
            output = output.squeeze(0) # num_classes
            probs = torch.softmax(output, 0)
            assert probs.sum() < 1.00001 and probs.sum() > .99999
            pred = torch.argmax(probs).item()
            y_pred.append(pred)
        return y_pred
