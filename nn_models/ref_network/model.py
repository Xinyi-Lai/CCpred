import torch
import torch.nn as nn
import torch.nn.functional as F


# CNN2-LSTM2
class CNN2_LSTM2(nn.Module):

    def __init__(self, seq_len, input_size):
        super(CNN2_LSTM2, self).__init__()
        # self.cnn_out_channels = cnn_out_channels
        # self.cnn_kernel_size = cnn_kernel_size
        # self.lstm_hidden_size = lstm_hidden_size
        # self.lstm_num_layers = lstm_num_layers
        self.conv1_out = 8
        self.conv2_out = 8
        self.lstm1_out = 4
        self.lstm2_out = 4
        # self.conv1_out = 64
        # self.conv2_out = 32
        # self.lstm1_out = 16
        # self.lstm2_out = 8
        
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels = input_size, 
                out_channels = self.conv1_out, 
                kernel_size = 3,
                padding='same'
            ),
            nn.ReLU(),
            nn.MaxPool1d( kernel_size = 2 ),
            nn.BatchNorm1d( num_features = self.conv1_out )
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(
                in_channels = self.conv1_out, 
                out_channels = self.conv2_out, 
                kernel_size = 3,
                padding='same'
            ),
            nn.ReLU(),
            nn.MaxPool1d( kernel_size = 2 ),
            nn.BatchNorm1d( num_features = self.conv2_out )
        )
        self.lstm1 = nn.LSTM(
            input_size = self.conv2_out,
            hidden_size = self.lstm1_out,
            num_layers =  1,
            batch_first = True,
            bidirectional = False
        )
        self.lstm2 = nn.LSTM(
            input_size = self.lstm1_out,
            hidden_size = self.lstm2_out,
            num_layers =  1,
            batch_first = True,
            bidirectional = False
        )
        self.fc = nn.Linear(self.lstm2_out, 1)

    def forward(self, x):
        #                x.shape: [batch_size, seq_len, num_features]
        x = x.permute(0,2,1)    # [batch_size, num_features, seq_len]
        x = self.conv1(x)       # [batch_size, conv1_out, seq_len/2]
        x = self.conv2(x)       # [batch_size, conv2_out, seq_len/4]
        x = x.permute(0,2,1)    # [batch_size, seq_len/4, conv2_out]
        x, _ = self.lstm1(x)    # [batch_size, seq_len/4, lstm1_out]
        x, _ = self.lstm2(x)    # [batch_size, seq_len/4, lstm2_out]
        x = x[:, -1, :]         # [batch_size, 1, lstm2_out] # seq_len != 1, take the last output from lstm
        x = self.fc(x)          # [batch_size, 1, 1]
        return x



# CNN-LSTM
class CNN_LSTM(nn.Module):

    def __init__(self, params):
        # params = { seq_len, input_size, cnn_out_channels, cnn_kernel_size, lstm_hidden_size, lstm_num_layers }
        super(CNN_LSTM, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels = params['input_size'], 
                out_channels = params['cnn_out_channels'], 
                kernel_size = params['cnn_kernel_size'],
                padding='same'
            ),
            nn.ReLU(),
            nn.MaxPool1d(
                kernel_size = params['cnn_kernel_size'], 
                stride=1
            )
        )
        self.bn = nn.BatchNorm1d(params['cnn_out_channels'])
        self.lstm = nn.LSTM(
            input_size = params['cnn_out_channels'],
            hidden_size = params['lstm_hidden_size'],
            num_layers =  params['lstm_num_layers'],
            batch_first = True,
            bidirectional = False
        )
        self.fc = nn.Linear(params['lstm_hidden_size'], 1)

    def forward(self, x):
        #                x.shape: [batch_size, seq_len, num_features]
        x = x.permute(0,2,1)    # [batch_size, num_features, seq_len]
        x = self.conv(x)        # [batch_size, cnn_out_channels, seq_len-kernel_size+1]
        x = self.bn(x)          # [batch_size, cnn_out_channels, seq_len-kernel_size+1]
        x = x.permute(0,2,1)    # [batch_size, seq_len-kernel_size+1, cnn_out_channels]
        x, _ = self.lstm(x)     # [batch_size, seq_len-kernel_size+1, hidden_size]
        x = x[:, -1, :]         # [batch_size, 1, hidden_size] # seq_len != 1, take the last output from lstm
        x = self.fc(x)          # [batch_size, 1, 1]
        return x
    

# LSTM
class LSTM(nn.Module):

    def __init__(self, input_size, lstm_hidden_size, lstm_num_layers):
        super(LSTM, self).__init__()
        self.lstm_hidden = lstm_hidden_size
        self.lstm_layers = lstm_num_layers
        self.lstm = nn.LSTM(
            input_size = input_size,
            hidden_size = self.lstm_hidden,
            num_layers =  self.lstm_layers,
            batch_first = True,
            bidirectional = False
        )
        self.fc = nn.Linear(self.lstm_hidden, 1)

    def forward(self, x):
        #                     x: [batch_size, seq_len, num_features]
        x, _ = self.lstm(x) # x: [batch_size, seq_len, hidden_size]
        x = x[:, -1, :]     # x: [batch_size, 1, hidden_size] # seq_len != 1, take the last output from lstm
        x = self.fc(x)      # x: [batch_size, 1, 1]
        return x



# class RNN_BI(nn.Module):
#
#     def __init__(self, input_dim, hidden_dim=50, batch_size=1, output_dim=1, num_layers=2, rnn_type='LSTM'):
#         super(RNN_BI, self).__init__()
#         self.input_dim = input_dim
#         self.hidden_dim = hidden_dim
#         self.batch_size = batch_size
#         self.num_layers = num_layers
#
#         #Define the initial linear hidden layer
#         self.init_linear = nn.Linear(self.input_dim, self.input_dim)
#
#         # Define the LSTM layer
#         self.lstm = eval('nn.' + rnn_type)(self.input_dim, self.hidden_dim, self.num_layers, batch_first=True, bidirectional=True)
#
#         # Define the output layer
#         self.linear = nn.Linear(self.hidden_dim * 2, output_dim)
#
#     def init_hidden(self):
#         # This is what we'll initialise our hidden state as
#         return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
#                 torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))
#
#     def forward(self, input):
#         #Forward pass through initial hidden layer
#         linear_input = self.init_linear(input)
#
#         # Forward pass through LSTM layer
#         # shape of lstm_out: [batch_size, input_size ,hidden_dim]
#         # shape of self.hidden: (a, b), where a and b both
#         # have shape (batch_size, num_layers, hidden_dim).
#         lstm_out, self.hidden = self.lstm(linear_input)
#
#         # Can pass on the entirety of lstm_out to the next layer if it is a seq2seq prediction
#         y_pred = self.linear(lstm_out)
#         return y_pred



# # RNN神经网络
# class RNN(nn.Module):
#     def __init__(self, input_size, hidden_size=50, output_size=1, num_layers=1):
#         super(RNN, self).__init__()
#         self.rnn = nn.RNN(input_size, hidden_size, num_layers)
#         self.reg = nn.Linear(hidden_size, output_size)

#     def forward(self, x):
#         x, _ = self.rnn(x) # 未在不同序列中传递hidden_state
#         return self.reg(x)

