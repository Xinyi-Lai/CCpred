""" Sequence to Sequence with encoder-decoder structure, 
    context vector to every step
"""

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import torch
import torch.nn as nn
import random

from models.nn_model import *
from models import Seq2Seq_model


class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size=32, hidden_size=64, n_layers=1, dropout=0.5):
        # NOTE: to use this structure, n_layers=1 for both encoder and decoder
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Linear(input_size, embedding_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.rnn = nn.GRU(embedding_size, hidden_size, n_layers)

    def forward(self, x):               # (Batch, in_seq_len, n_in_feat)
        x = x.permute(1,0,2)            # (in_seq_len, Batch, n_in_feat)
        embedded = self.embedding(x)    # (in_seq_len, Batch, embedding_size)  
        output, hidden = self.rnn(embedded)
        # output, (in_seq_len, Batch, hidden_size): output of the last layer for each time step
        # hidden, (n_layers, Batch, hidden_size): final hidden state of each layer, the "context" used to feed into the decoder
        return hidden


class Decoder(nn.Module):
    def __init__(self, output_size=1, embedding_size=32, hidden_size=64, n_layers=1, dropout=0.5):
        # NOTE: to use this structure, n_layers=1 for both encoder and decoder
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Linear(output_size, embedding_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.rnn = nn.GRU(embedding_size+hidden_size, hidden_size, n_layers)
        self.fc = nn.Linear(embedding_size+2*hidden_size, output_size)

    def forward(self, x, hidden, context):  
        # x: (Batch, n_out_feat), NOTE: only 2 dimensions here, out_seq_len is removed because we are mannualy performing rnn
        # hidden:  (n_layers, Batch, hidden_size), n_layers=1, hidden state of rnn in decoder, change at each step
        # context: (n_layers, Batch, hidden_size), n_layers=1, output hidden state from encoder, same feed at each step
                                        
        # add sequence dimension to x, to allow the use of rnn
        x = x.unsqueeze(0)              # (1, Batch, output_size)
        embedded = self.embedding(x)    # (1, Batch, embedding_size)
        embedded_cat = torch.cat((embedded, context), dim=2) # (1, Batch, embedding_size + hidden_size)

        # seq_len, n_layers, and n_directions are 1 in the decoder, therefore:
        # output: (1, Batch, hidden_size)
        # hidden: (1, Batch, hidden_size)
        output, hidden = self.rnn(embedded_cat, hidden)

        output = torch.cat((embedded, hidden, context), dim=2).squeeze(0) # (Batch, embedding_size + 2*hidden_size)
        prediction = self.fc(output)    # (Batch, output_size)

        return prediction, hidden


class Seq2SeqPlus(nn.Module):
    def __init__(self, input_size, output_size, device, embedding_size=32, hidden_size=64):
        super().__init__()
        self.encoder = Encoder(input_size, embedding_size, hidden_size).to(device)
        self.decoder = Decoder(output_size, embedding_size, hidden_size).to(device)
        self.device = device
        
    def forward(self, x, y, teacher_forcing_ratio=0.5):
        """
            x: (Batch, in_seq_len, n_in_feat)
            y: (Batch, out_seq_len, n_out_feat)
            teacher_forcing_ratio: the probability of using teacher forcing (using ground-truth inputs)
        """

        target_len = y.shape[1]
        # tensor to store decoder outputs of each time step
        outputs = torch.zeros(y.shape).to(self.device) # (Batch, out_seq_len)
        # the last hidden state of the encoder is used as the context
        context = self.encoder(x)
        # the last hidden state of the encoder is also used as the initial hidden state of the decoder
        hidden = context
        # first input to decoder is the last observed value #FIXME# hardcode
        decoder_input = x[:, -1, 0:1] # (Batch, out_feature_size)
        for i in range(target_len):
            # run decode for each time step
            output, hidden = self.decoder(decoder_input, hidden, context)
            # place predictions to a tensor holding predictions for each time step
            outputs[:,i] = output
            # decide if we are going to use teacher forcing or not
            teacher_forcing = random.random() < teacher_forcing_ratio
            # if teacher forcing, use ground truth as next input; if not, use predicted output
            decoder_input = y[:,i,:] if teacher_forcing else output

        return outputs



class Seq2SeqPlus_model(Seq2Seq_model):

    def __init__(self, model_name='Seq2SeqPlus', batch_size=1) -> None:
        super().__init__(model_name, batch_size)

    def init_model(self, embedding_size=32, hidden_size=64):
        self.model = Seq2SeqPlus(self.n_in_feat, 1, self.device, embedding_size, hidden_size).to(self.device)
        return

    # the following methods are inherited from Seq2Seq_model:
    # prepare_data, train_model, get_forecast, vis_performance