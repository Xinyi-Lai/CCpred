""" Sequence to Sequence
"""

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import torch
import torch.nn as nn
import random

from nn_models.nn_model import *
from nn_models import Seq2Seq_model


class Encoder(nn.Module):
    def __init__(self, input_size, emb_size=16, enc_hid_size=32, dec_hid_size=32, dropout=0.5):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Linear(input_size, emb_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.rnn = nn.GRU(emb_size, enc_hid_size, bidirectional=True) # 1 layer, bidirectional
        self.act = nn.Sequential(
            nn.Linear(enc_hid_size*2, dec_hid_size),
            nn.Tanh()
        )

    def forward(self, x):
        """ Args: x: (Batch, in_seq_len, n_in_feat), input sequence
            Rets: s: (Batch, dec_hid_size), final hidden state (foreward and backward) of the last layer
                  output: (in_seq_len, Batch, enc_hid_size*2), output of the last layer for each time step
        """
        x = x.permute(1,0,2)        # (in_seq_len, Batch, n_in_feat)
        embedded = self.embedding(x) #(in_seq_len, Batch, emb_size)  
        
        # output: (in_seq_len, Batch, enc_hid_size*2), output of the last layer for each time step, bidirectional
        # hidden: (n_layers*num_dir(=2), Batch, enc_hid_size), final hidden state of each layer
        #           stacked as [foreward_0, backward_0, foreward_1, backward_1, ...]
        output, hidden = self.rnn(embedded)
        
        cat = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1) # (Batch, enc_hid_size*2)
        s = self.act(cat) # (Batch, dec_hid_size), final hidden state (foreward and backward) of the last layer

        return s, output



class Attention(nn.Module):
    def __init__(self, enc_hid_size=32, dec_hid_size=32):
        super().__init__()
        self.attn = nn.Linear(enc_hid_size*2+dec_hid_size, dec_hid_size, bias=False)
        self.v = nn.Linear(dec_hid_size, 1, bias=False)

    def forward(self, s, enc_out):
        """ Args: s: (Batch, dec_hid_size), decoder hidden state
                  enc_out: (in_seq_len, Batch, enc_hid_size*2), encoder output
            Rets: a: (Batch, in_seq_len), attention weights
        """
        # repeat hidden state s to match the shape of enc_out # FIXME use hidden states of all steps?
        in_seq_len = enc_out.shape[0]
        s = s.unsqueeze(0).repeat(in_seq_len, 1, 1) # (in_seq_len, Batch, dec_hid_size)
        a = torch.cat((s, enc_out), dim=2)          # (in_seq_len, Batch, dec_hid_size + enc_hid_size*2), [s_(t-1), H]
        a = torch.tanh(self.attn(a))                # (in_seq_len, Batch, dec_hid_size), E_t
        a = self.v(a).squeeze(2)                    # (in_seq_len, Batch), a'_t
        a = torch.softmax(a, dim=1)                 # (in_seq_len, Batch), a_t
        return a.permute(1,0)                       # (Batch, in_seq_len)



class Decoder(nn.Module):
    def __init__(self, attention, output_size=1, emb_size=16, enc_hid_size=32, dec_hid_size=32, dropout=0.5):
        super().__init__()
        self.attention = attention
        self.embedding = nn.Sequential(
            nn.Linear(output_size, emb_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.rnn = nn.GRU(enc_hid_size*2+emb_size, dec_hid_size) # 1 layer, unidirectional
        self.fc = nn.Linear(enc_hid_size*2+dec_hid_size+emb_size, output_size)

    def forward(self, dec_in, s, enc_out):
        """ Args: dec_in: (Batch, n_out_feat), NOTE: only 2 dimensions here, out_seq_len is removed because we are mannualy performing rnn
                  s: (Batch, dec_hid_size), hidden state of rnn at the previous step
                  enc_out: (in_seq_len, Batch, enc_hid_size*2), output of encoder
            Rets: pred: (Batch, n_out_feat), output of the last layer for each time step
                  s: (Batch, output_size), updated hidden state (foreward and backward) of the last layer
        """
        # calculate context with attention
        att = self.attention(s, enc_out).unsqueeze(1)   # (Batch, 1, in_seq_len)
        enc_out = enc_out.permute(1,0,2)                # (Batch, in_seq_len, enc_hid_dim*2)
        ctx = torch.bmm(att, enc_out).permute(1,0,2)    # (1, Batch, enc_hid_dim*2)

        # augment embedded with context, then rnn
        dec_in = dec_in.unsqueeze(0)        # (1, Batch, output_size), add sequence dimension to x, to allow the use of rnn
        embedded = self.embedding(dec_in)   # (1, Batch, emb_size)
        rnn_in = torch.cat((embedded, ctx), dim = 2) # (1, Batch, emb_size+enc_hid_dim*2)
        # dec_out: (seq_len(=1), Batch, dec_hid_size)
        # dec_hid: (n_layers*num_dir(=1), Batch, dec_hid_size)
        dec_out, dec_hid = self.rnn(rnn_in, s.unsqueeze(0))

        # augment dec_out with context and embedded, then predict with fc
        cat = torch.cat((dec_out, ctx, embedded), dim = 2) # (1, Batch, dec_hid_size + enc_hid_size*2 + emb_size)
        pred = self.fc(cat.squeeze(0))      # (Batch, output_size)
        
        return pred, dec_hid.squeeze(0)



class Seq2SeqAtt(nn.Module):
    def __init__(self, input_size, output_size, device, emb_size=16, hid_size=32):
        super().__init__() # TODO
        self.encoder = Encoder(input_size, emb_size, hid_size, hid_size).to(device)
        self.attention = Attention(hid_size, hid_size).to(device)
        self.decoder = Decoder(self.attention, output_size, emb_size, hid_size, hid_size).to(device)
        self.device = device
        
    def forward(self, x, y, teacher_forcing_ratio=0.5):
        """ x: source, (Batch, in_seq_len, n_in_feat)
            y: target, (Batch, out_seq_len, n_out_feat)
            teacher_forcing_ratio: the probability of using teacher forcing (using ground-truth inputs)
        """
        target_len = y.shape[1]

        # tensor to store decoder outputs of each time step
        outputs = torch.zeros(y.shape).to(self.device) # (Batch, out_seq_len)
        
        # s, (Batch, hid_size): final hidden states of the last layer after activation, used as the initial hidden state of the decoder                     
        # enc_out, (in_seq_len, Batch, hid_size*2): output of the last layer for all steps
        s, enc_out = self.encoder(x)

        # first input to decoder is the last observed value #FIXME# hardcoded
        dec_in = x[:, -1, 0:1] # (Batch, out_feature_size)

        for i in range(target_len):
            # run decode for each time step
            dec_out, s = self.decoder(dec_in, s, enc_out)
            # place predictions to a tensor holding predictions for each time step
            outputs[:,i] = dec_out
            # decide if we are going to use teacher forcing or not
            teacher_forcing = random.random() < teacher_forcing_ratio
            # if teacher forcing, use ground truth as next input; if not, use predicted output
            dec_in = y[:,i] if teacher_forcing else dec_out

        return outputs



class Seq2SeqAtt_model(Seq2Seq_model):

    def __init__(self, model_name='Seq2SeqAtt', batch_size=1) -> None:
        super().__init__(model_name, batch_size)

    def init_model(self, embedding_size=16, hidden_size=32):
        self.model = Seq2SeqAtt(self.n_in_feat, 1, self.device, embedding_size, hidden_size).to(self.device)
        return

    # the following methods are inherited from Seq2Seq_model:
    # prepare_data, train_model, get_forecast, vis_performance