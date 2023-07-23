""" Sequence to Sequence with encoder-decoder structure
"""

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import torch
import torch.nn as nn
import random

from models.nn_model import *


class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size=32, hidden_size=64, n_layers=2, dropout=0.5):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Linear(input_size, embedding_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.rnn = nn.GRU(embedding_size, hidden_size, n_layers, dropout=dropout)

    def forward(self, x):               # (Batch, in_seq_len, n_in_feat)
        x = x.permute(1,0,2)            # (in_seq_len, Batch, n_in_feat)
        embedded = self.embedding(x)    # (in_seq_len, Batch, embedding_size)  
        output, hidden = self.rnn(embedded)
        # output, (in_seq_len, Batch, hidden_size): output of the last layer for each time step
        # hidden, (n_layers, Batch, hidden_size): final hidden state of each layer, the "context" used to feed into the decoder
        return hidden


class Decoder(nn.Module):
    def __init__(self, output_size=1, embedding_size=32, hidden_size=64, n_layers=2, dropout=0.5):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Linear(output_size, embedding_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.rnn = nn.GRU(embedding_size, hidden_size, n_layers, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):       # x: (Batch, n_out_feat), NOTE: only 2 dimensions here, out_seq_len is removed because we are mannualy performing rnn
                                        # hidden: (n_layers, Batch, hidden_size)
        # add sequence dimension to x, to allow the use of rnn
        x = x.unsqueeze(0)              # (1, Batch, output_size)
        embedded = self.embedding(x)    # (1, Batch, embedding_size)

        # seq_len and n_directions is 1 in the decoder, therefore:
        # output: (Batch, 1, hidden_size)
        # hidden: (n_layers, Batch, hidden_size)
        output, hidden = self.rnn(embedded, hidden)

        output = output.squeeze(0)      # (Batch, hidden_size)
        prediction = self.fc(output)    # (Batch, output_size)

        return prediction, hidden


class Seq2Seq(nn.Module):
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
        # last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden = self.encoder(x)
        # first input to decoder is the last observed value #FIXME#
        decoder_input = x[:, -1, 0:1] # (Batch, out_feature_size)
        for i in range(target_len):
            # run decode for each time step
            output, hidden = self.decoder(decoder_input, hidden)
            # place predictions to a tensor holding predictions for each time step
            outputs[:,i] = output
            # decide if we are going to use teacher forcing or not
            teacher_forcing = random.random() < teacher_forcing_ratio
            # if teacher forcing, use ground truth as next input; if not, use predicted output
            decoder_input = y[:,i,:] if teacher_forcing else output

        return outputs



class Seq2Seq_model(NN_model):

    def __init__(self, model_name='Seq2Seq', batch_size=1) -> None:
        super().__init__(model_name, batch_size)


    def prepare_data(self, dataX, dataY, seq_len=30, pred_len=1):
        """ organize and store dataset for nn model
            NOTE: assume dataX and dataY are aligned
            Args:
                dataX (np.array): features, size of (win_len, n_in_feat), n_in_feat = n_x_var.
                dataY (np.array): labels, size of (win_len, n_out_feat), n_out_feat = 1.
                seq_len (int, optional): input sequence length, not win_len. Defaults to 30.
                pred_len (int, optional): output sequence length, num of steps to predict. Defaults to 1.
        """
        win_len, n_in_feat = dataX.shape
        win_len_y, n_out_feat = dataY.shape
        assert win_len ==  win_len_y, "win_len_x and win_len_y do not match"
        self.n_in_feat = n_in_feat
        self.n_out_len = pred_len

        # normalize as columns
        scalarX = StandardScaler() # StandardScaler() # MinMaxScaler()
        dataX = scalarX.fit(dataX).transform(dataX)
        scalarY = StandardScaler() # StandardScaler() # MinMaxScaler()
        dataY = scalarY.fit(dataY).transform(dataY)
        self.scalar = scalarY
        
        # make dataset
        dataset = []
        for i in range(win_len - seq_len - pred_len + 1):
            x = torch.FloatTensor(dataX[i : i+seq_len, :]).to(self.device) # (seq_len, n_in_feat)
            y = torch.FloatTensor(dataY[i+seq_len : i+seq_len+pred_len, :]).to(self.device) # (pred_len, n_out_feat)
            dataset.append((x,y))
        last_seq = dataX[-seq_len:, :] # (seq_len, n_in_feat)
        self.last_seq = torch.FloatTensor(last_seq[np.newaxis, :]).to(self.device) # (1, seq_len, n_comp), add a batch dimension 
        self.dataset = dataset

        return


    def init_model(self, embedding_size=32, hidden_size=64):
        self.model = Seq2Seq(self.n_in_feat, 1, self.device, embedding_size, hidden_size).to(self.device)
        return

    
    def train_model(self):
        """ seq2seq has a slightly different training process from other models
        """
        # split datasets with shuffling (sequences are independent, so it's ok to shuffle)
        split = int(len(self.dataset)*0.8)
        Dtr, Dte = random_split(self.dataset, [split, len(self.dataset)-split])
        Dtr = DataLoader(MyDataset(Dtr), batch_size=self.batch_size)
        Dte = DataLoader(MyDataset(Dte), batch_size=self.batch_size)
        
        # warm start if applicable
        model_path = os.path.join('saved_model', self.model_name, 'best_model.pth')
        try:
            if os.path.exists(model_path):
                self.model.load_state_dict(torch.load(model_path))
        except Exception as e:
            print('error in warm start: ',e.__class__.__name__,e)
        
        # define loss and optimizer
        loss_func = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        # initialize an early-stopping instance
        model_dir = os.path.join('saved_model', self.model_name)
        early_stopping = EarlyStopping(model_dir)
        
        for e in range(100):
            # training
            train_loss = []
            self.model.train()
            for seq, label in Dtr:
                y_pred = self.model(seq, label)
                loss = loss_func(y_pred, label)
                train_loss.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            train_loss = np.mean(train_loss)

            # validate
            val_loss = []
            self.model.eval() # testing mode
            with torch.no_grad(): # no backward propagation, no computational graph
                for seq, label in Dte:
                    # turn off teacher forcing !!!
                    y_pred = self.model(seq, label, teacher_forcing_ratio=0)
                    loss = loss_func(y_pred, label)
                    val_loss.append(loss.item())
            val_loss = np.mean(val_loss)
            
            print('epoch {:03d} train_loss {:.8f} val_loss {:.8f}'.format(e+1, train_loss, val_loss))

            # if there is no improvement beyond some patience, 'early_stop' is set to True
            early_stopping(val_loss, self.model)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        
        return


    def get_forecast(self):
        """ seq2seq has a slightly different forecasting process from other models
        """
        # a null teacher
        ynull = torch.zeros((self.batch_size, self.n_out_len, 1)).to(self.device)
        self.pred = self.model(self.last_seq, ynull, teacher_forcing_ratio=0).cpu().detach().numpy() # (1,n_out)
        self.pred = self.scalar.inverse_transform(self.pred.reshape(-1,1)).reshape(-1)
        return


    def vis_performance(self, plot=False):
        """ visualize model fitting performance, use the whole dataset
            metrics (rmse and mape) of each columns are printed, the first 4 rows are plotted
        """
        Dtr = DataLoader(MyDataset(self.dataset), batch_size=self.batch_size)
        real = []
        pred = []
        self.model.eval()
        with torch.no_grad():
            for seq, label in Dtr: # use the whole ordered dataset for testing and visualization
                ypred = self.model(seq, label, teacher_forcing_ratio=0)
                real.extend(label.cpu().numpy())
                pred.extend(ypred.cpu().numpy())
        pred = self.scalar.inverse_transform(np.array(pred).reshape(-1, self.n_out_len))
        real = self.scalar.inverse_transform(np.array(real).reshape(-1, self.n_out_len))

        return

    def apply_model(self, plot=False):
        """ apply model on the whole dataset, visualize model fitting performance 
            metrics (rmse and mape) of each columns are printed, the first 4 rows are plotted
        """
        Dtr = DataLoader(MyDataset(self.dataset), batch_size=self.batch_size)
        real = []
        pred = []
        self.model.eval()
        with torch.no_grad():
            for seq, label in Dtr: # use the whole ordered dataset for testing and visualization
                ypred = self.model(seq, label, teacher_forcing_ratio=0)
                real.extend(label.cpu().numpy())
                pred.extend(ypred.cpu().numpy())
        pred = self.scalar.inverse_transform(np.array(pred).reshape(-1, self.n_out_len))
        real = self.scalar.inverse_transform(np.array(real).reshape(-1, self.n_out_len))

        show_performance(self.model_name, pred, real, plot)
        return pred, real