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

    def __init__(self, model_name='BPNN', batch_size=1) -> None:
        super().__init__(model_name, batch_size)


    def prepare_data(self, dataX, dataY, seq_len=30, pred_len=1):
        """ organize and store dataset for nn model
            FUTURE: dataX and dataY are aligned, we manually reorganize x and y here, to be revised in future
            Args:
                dataX (np.array): features, size of (win_len, n_comp)
                dataY (np.array): labels, size of (win_len,)
                seq_len (int, optional): sequence length in TCN. Defaults to 30.
                pred_len (int, optional): num of steps to predict. Defaults to 1.
        """
        win_len, n_comp = dataX.shape
        dataY = dataY[:, np.newaxis]
        self.n_in = seq_len*n_comp # TODO
        self.n_out = pred_len

        # normalize as columns
        scalarX = MinMaxScaler()
        dataX = scalarX.fit(dataX).transform(dataX)
        scalarY = MinMaxScaler()
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


    def init_model(self, hidden_size=50):
        self.model = BPNN(input_size=self.n_in, hidden_size=hidden_size, output_size=self.n_out).to(self.device)
        return
