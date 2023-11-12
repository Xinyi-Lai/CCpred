""" Multi-Layer Perceptron
"""

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import torch

from models.nn_model import NN_model


class MLP(torch.nn.Module):
    """ MLP
        Args:
            input_size (int): num of input features (flattened).
            hidden_size (int): a list of hidden_size of each layer, e.g., hidden_size=[25,25] means 2 hidden layers with 25 cells in each layer
            output_size (int, optional): num of output channels. Defaults to 1.
    """
    def __init__(self, input_size, hidden_size, output_size=1):
        super(MLP, self).__init__()
        layers = []
        num_layers = len(hidden_size)
        for i in range(num_layers):
            if i==0:
                layers += [torch.nn.Linear(input_size, hidden_size[i])]
            else:
                layers += [torch.nn.Linear(hidden_size[i-1], hidden_size[i])]
            layers += [torch.nn.ReLU()]
            # layers += [torch.nn.Dropout(p=0.1)]
        
        self.mlp = torch.nn.Sequential(*layers)
        self.predict = torch.nn.Linear(hidden_size[-1], output_size)
    def forward(self, x):   # x: (Batch, seq_len+n_comp-1)
        x = self.mlp(x)     # x: (Batch, hidden_size[-1])
        x = self.predict(x) # x: (Batch, output_size)
        return x



class MLP_model(NN_model):

    def __init__(self, model_name='MLP', batch_size=1, lr=0.001) -> None:
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
        self.in_n = seq_len+n_comp-1 # num of input features + num of prev steps (current step is counted twice)
        self.out_n = pred_len
        self.in_dim = (self.batch_size,  seq_len+n_comp-1)  # shape of input to network

        # normalize as columns
        scalarX = StandardScaler() # StandardScaler() # MinMaxScaler()
        dataX = scalarX.fit(dataX).transform(dataX)
        scalarY = StandardScaler() # StandardScaler() # MinMaxScaler()
        dataY = scalarY.fit(dataY).transform(dataY)
        self.scalar = scalarY
        
        # make dataset
        dataset = []
        for i in range(win_len-seq_len-pred_len+1):
            x = list(dataX[i:i+seq_len, 0]) + list(dataX[i+seq_len-1, 1:])
            x = torch.FloatTensor( np.array(x).reshape(-1) ).to(self.device) # (seq_len+n_comp-1,)
            y = torch.FloatTensor(dataY[i+seq_len : i+seq_len+pred_len].reshape(-1)).to(self.device) # (pred_len,)
            dataset.append((x,y))
        last_seq = np.array( list(dataX[-seq_len:, 0]) + list(dataX[-1, 1:]) ).reshape(-1) # (seq_len+n_comp-1,)
        self.last_seq = torch.FloatTensor(last_seq[np.newaxis, :]).to(self.device) # (1, seq_len+n_comp-1), add a batch dimension 
        self.dataset = dataset

        return


    def init_model(self, hidden_size=[25,10]):
        self.model = MLP(self.in_n, hidden_size, self.out_n).to(self.device)
        return
