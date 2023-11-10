""" Back Propagation Neural Network
"""

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import torch

from models.nn_model import NN_model


class BPNN(torch.nn.Module):
    """ BPNN
        Args:
            input_size (int): num of input features (flattened).
            hidden_size (int): number of features in the hidden layer.
            output_size (int, optional): num of output channels. Defaults to 1.
    """
    def __init__(self, input_size, hidden_size, output_size=1):
        super(BPNN, self).__init__()
        self.hidden = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.predict = torch.nn.Linear(hidden_size, output_size)
    def forward(self, x):   # x: (Batch, seq_len*n_comp)
        x = self.hidden(x)  # x: (Batch, hidden_size)
        x = self.relu(x)    # x: (Batch, hidden_size)
        x = self.predict(x) # x: (Batch, output_size)
        return x



class BPNN_model(NN_model):

    def __init__(self, model_name='BPNN', batch_size=1, lr=0.005) -> None:
        super().__init__(model_name, batch_size, lr)


    def prepare_data(self, dataX, dataY, seq_len=30, pred_len=1):
        """ organize and store dataset for nn model
            NOTE: assume dataX and dataY are aligned
            Args:
                dataX (np.array): features, size of (win_len, n_comp)
                dataY (np.array): labels, size of (win_len, 1)
                seq_len (int, optional): sequence length. Defaults to 30.
                pred_len (int, optional): num of steps to predict. Defaults to 1.
        """
        win_len, n_comp = dataX.shape
        self.in_n = seq_len*n_comp
        self.out_n = pred_len
        self.in_dim = (self.batch_size,  seq_len*n_comp)  # shape of input to network

        # normalize as columns
        scalarX = StandardScaler() # StandardScaler() # MinMaxScaler()
        dataX = scalarX.fit(dataX).transform(dataX)
        scalarY = StandardScaler() # StandardScaler() # MinMaxScaler()
        dataY = scalarY.fit(dataY).transform(dataY)
        self.scalar = scalarY
        
        # make dataset
        dataset = []
        for i in range(win_len-seq_len-pred_len+1):
            x = torch.FloatTensor(dataX[i : i+seq_len, :].reshape(-1)).to(self.device) # (seq_len*n_comp,)
            y = torch.FloatTensor(dataY[i+seq_len : i+seq_len+pred_len].reshape(-1)).to(self.device) # (pred_len,)
            dataset.append((x,y))
        last_seq = dataX[-seq_len:, :].reshape(-1) # (seq_len*n_comp,)
        self.last_seq = torch.FloatTensor(last_seq[np.newaxis, :]).to(self.device) # (1, seq_len*n_comp), add a batch dimension 
        self.dataset = dataset

        return


    def init_model(self, hidden_size=10):
        self.model = BPNN(input_size=self.in_n, hidden_size=hidden_size, output_size=self.out_n).to(self.device)
        return
