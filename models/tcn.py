""" Temporal Convolutional Network
    Bai, Shaojie, J. Zico Kolter and Vladlen Koltun. “An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling.” ArXiv abs/1803.01271 (2018): n. pag.
    https://arxiv.org/pdf/1803.01271.pdf
    http://github.com/locuslab/TCN
"""

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

from models.nn_model import NN_model


class Chomp1d(nn.Module):
    """ a cropping block, chomp the extra padding
        Args:
            chomp_size (int): size of padding to be cropped
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """ a residual block
        Args:
            n_inputs (int): num of input features
            n_outputs (int): num of output features
            kernel_size (int): kernel size in convolution
            dilation (int): dilation coeff
            padding (int): padding coeff
            stride (int): stride in convolution. Defaults to 1
            dropout (float, optional): dropout coeff. Defaults to 0.2.
    """
    def __init__(self, n_inputs, n_outputs, kernel_size, dilation, padding, stride=1, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.net = nn.Sequential(   self.conv1,         # x: (Batch, n_outputs, seq_len + padding)
                                    Chomp1d(padding),   # x: (Batch, n_outputs, seq_len)
                                    nn.ReLU(),          # x: (Batch, n_outputs, seq_len)
                                    nn.Dropout(dropout),# x: (Batch, n_outputs, seq_len)
                                    self.conv2,         # x: (Batch, n_outputs, seq_len + padding)
                                    Chomp1d(padding),   # x: (Batch, n_outputs, seq_len)
                                    nn.ReLU(),          # x: (Batch, n_outputs, seq_len)
                                    nn.Dropout(dropout))# x: (Batch, n_outputs, seq_len)
        # if input size (residual input) and output size (conv output) don't agree, add a conv 
        # layer to transform the channels, so that they can be added together and fed into relu
        if n_inputs != n_outputs:   self.downsample = nn.Conv1d(n_inputs, n_outputs, 1)
        else:                       self.downsample = None
        self.relu = nn.ReLU()
        self.init_weights()
        
    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):                   # x:   (Batch, n_inputs, seq_len)
        out = self.net(x)                   # out: (Batch, n_outputs, seq_len)
        if self.downsample is None: res = x # res: (Batch, n_outputs(=n_inputs), seq_len)
        else: res = self.downsample(x)      # res: (Batch, n_outputs, seq_len)
        return self.relu(out + res)         #      (Batch, n_outputs, seq_len)


class TemporalConvNet(nn.Module):
    """ temporal convolutional network
        the current structure supports 1D sequence modeling well, i.e., num_inputs = 1;
        for vector series (a vector input at each timestep), take it as different input channels;
        for matrix series (e.g. an image input at each timestep), it would be kinda sticky
        Args:
            num_inputs (int): num of input features
            num_channels (list): a list of num_channel of each layer, e.g., num_channels=[25,25,25,25] means 4 hidden layers with 25 hidden channels in each layer
            num_outputs (int, optional): num of output features. Defaults to 1.
            kernel_size (int, optional): kernel size in convolution. Defaults to 2.
            dropout (float, optional): dropout coeff. Defaults to 0.2.
    """
    def __init__(self, num_inputs, num_channels, num_outputs=1, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i   # dilation = 1, 2, 4, 8...
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, stride=1, dropout=dropout)]
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], num_outputs)

    def forward(self, x):   # (Batch, num_inputs, seq_len]
        # NOTE: A bit tricky here. 
        # For RNNs, input size is usually (Batch, seq_len, channels) or (seq_len, Batch, channels). 
        # Here, input size is (Batch, num_inputs, seq_len), so that it can convolve across timesteps.
        x = self.network(x) # (Batch, num_channels[-1], seq_len)
        x = x[:,:,-1]       # (Batch, num_channels[-1]) # the last of the sequence
        x = self.fc(x)      # (Batch, num_outputs)
        return x



class TCN_model(NN_model):

    def __init__(self, model_name='TCN', batch_size=1) -> None:
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
        self.n_in = n_comp
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
            x = torch.FloatTensor(dataX[i : i+seq_len, :].T).to(self.device) # (n_comp, seq_len)
            y = torch.FloatTensor(dataY[i+seq_len : i+seq_len+pred_len].reshape(-1)).to(self.device) # (pred_len,)
            dataset.append((x,y))
        last_seq = dataX[-seq_len:, :].T # (n_comp, seq_len)
        self.last_seq = torch.FloatTensor(last_seq[np.newaxis, :]).to(self.device) # (1, n_comp, seq_len), add a batch dimension 
        self.dataset = dataset

        return


    def init_model(self, n_channels=[10,25,10]):
        self.model = TemporalConvNet(num_inputs=self.n_in, num_channels=n_channels, num_outputs=self.n_out).to(self.device)
        return
