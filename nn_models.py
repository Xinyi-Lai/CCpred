'''
Neural network models for forecasting
!!! deprecated, see models.py
'''

import os
import numpy as np
import matplotlib.pyplot as plt 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from early_stopping import EarlyStopping


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# future: warp into DataLoader: done
class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __getitem__(self, item):
        return self.data[item]
    def __len__(self):
        return len(self.data)


# function of model training and forecast
# applicable to BPNN, LSTM, GRU and TCN, possible to more. 
def train_and_forecast(model, model_dir, Dtr, Dte, last_seq):
    # warm start if applicable
    model_path = os.path.join(model_dir, 'best_model.pth')
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
    # define loss and optimizer
    loss_func = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # initialize an early-stopping instance
    early_stopping = EarlyStopping(model_dir)
    
    epochs = 100
    for e in range(epochs):
        # training
        train_loss = []
        model.train()
        for seq, label in Dtr:
            y_pred = model(seq)
            loss = loss_func(y_pred, label)
            train_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_loss = np.mean(train_loss)

        # validate
        val_loss = []
        model.eval() # testing mode
        with torch.no_grad(): # no backward propagation, no computational graph
            for seq, label in Dte:
                y_pred = model(seq)
                loss = loss_func(y_pred, label)
                val_loss.append(loss.item())
        val_loss = np.mean(val_loss)
        
        print('epoch {:03d} train_loss {:.8f} val_loss {:.8f}'.format(e+1, train_loss, val_loss))

        # if there is no improvement beyond some patience, 'early_stop' is set to True
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    
    # forecasting        
    return model(last_seq).cpu().detach().numpy()[0], model



###### 4.1 with BPNN

### define model structure
class BPNN(torch.nn.Module):
    def __init__(self, n_features, n_hidden=50, n_output=1):
        super(BPNN, self).__init__()
        self.hidden = torch.nn.Linear(n_features, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)
    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x

### prepare dataset
def prepare_data_BPNN(series, seq_len, random=True):
    
    # normalize as a column
    scalar = MinMaxScaler()
    scalar.fit(series.reshape(-1,1))
    series = scalar.transform(series.reshape(-1,1)).reshape(-1)
    
    # make dataset
    Dtr = []
    for i in range(len(series)-seq_len):
        x = torch.FloatTensor(series[i:i+seq_len]).to(device)
        y = torch.FloatTensor([series[i+seq_len]]).to(device)
        Dtr.append((x,y))
    last_seq = torch.FloatTensor(series[-seq_len:]).to(device)
    
    # no shuffling (sequences are independent, it's ok to shuffle)
    # Dte, Dtr = MyDataset(Dtr[int(len(Dtr)*0.8):]), MyDataset(Dtr[:int(len(Dtr)*0.8)])
    Dtr = MyDataset(Dtr)
    # shuffle for training
    if random: Dtr, Dte = random_split(Dtr, [int(len(Dtr)*0.8), int(len(Dtr)*0.2)])
    # no shuffle for testing and visualization
    else: Dte = None

    return Dtr, Dte, last_seq, scalar
    
### model training and forecast
def forecast_BPNN(series, trail_name='BPNN', seq_len=30):
    Dtr, Dte, last_seq, scalar = prepare_data_BPNN(series, seq_len, True)
    model = BPNN(n_features=seq_len).to(device)
    model_dir = './saved_model/%s'%trail_name # for early stopping and warm start
    pred, model = train_and_forecast(model, model_dir, Dtr, Dte, last_seq)
    pred = scalar.inverse_transform(pred.reshape(-1,1)).reshape(-1)[0]
    return pred, model



###### 4.2 with LSTM / GRU

### define model structure
class LSTM(torch.nn.Module):
    def __init__(self, input_size=1, hidden_size=10, output_size=1):
        super(LSTM, self).__init__()
        self.lstm = torch.nn.LSTM( input_size = input_size, hidden_size = hidden_size )
        self.fc = torch.nn.Linear( hidden_size, output_size )
    def forward(self, x):   # x: [seq_len, num_features]                    
        x, _ = self.lstm(x) # x: [seq_len, hidden_size]
        x = x[-1, :]        # x: [1, hidden_size] # seq_len != 1, take the last output from lstm
        x = self.fc(x)      # x: [1, 1]
        return x

class GRU(torch.nn.Module):
    def __init__(self, input_size=1, hidden_size=10, output_size=1):
        super(GRU, self).__init__()
        self.gru = torch.nn.GRU( input_size = input_size, hidden_size = hidden_size )
        self.fc = torch.nn.Linear( hidden_size, output_size )
    def forward(self, x):   # x: [seq_len, num_features]                    
        x, _ = self.gru(x)  # x: [seq_len, hidden_size]
        x = x[-1, :]        # x: [1, hidden_size] # seq_len != 1, take the last output from lstm
        x = self.fc(x)      # x: [1, 1]
        return x

### prepare dataset (LSTM and GRU share the same process)
def prepare_data_LSTM_GRU(series, seq_len, random=True):
    
    # normalize as a column
    scalar = MinMaxScaler()
    scalar.fit(series.reshape(-1,1))
    series = scalar.transform(series.reshape(-1,1)).reshape(-1)
    
    # make dataset
    Dtr = []
    for i in range(len(series)-seq_len):
        x = torch.FloatTensor(series[i:i+seq_len, np.newaxis]).to(device) # [seq_len, input_channel]
        y = torch.FloatTensor([series[i+seq_len]]).to(device)
        Dtr.append((x,y))
    last_seq = torch.FloatTensor(series[-seq_len:, np.newaxis]).to(device) # [seq_len, input_channel]
    
    # no shuffling (sequences are independent, it's ok to shuffle)
    # Dte, Dtr = MyDataset(Dtr[int(len(Dtr)*0.8):]), MyDataset(Dtr[:int(len(Dtr)*0.8)])
    Dtr = MyDataset(Dtr)
    # shuffle for training
    if random: Dtr, Dte = random_split(Dtr, [int(len(Dtr)*0.8), int(len(Dtr)*0.2)])
    # no shuffle for testing and visualization
    else: Dte = None

    return Dtr, Dte, last_seq, scalar

### model training and forecast
def forecast_LSTM(series, trail_name='LSTM', seq_len=30):
    Dtr, Dte, last_seq, scalar = prepare_data_LSTM_GRU(series, seq_len, True)
    model = LSTM().to(device)
    model_dir = './saved_model/%s'%trail_name # for early stopping and warm start
    pred, model = train_and_forecast(model, model_dir, Dtr, Dte, last_seq)
    pred = scalar.inverse_transform(pred.reshape(-1,1)).reshape(-1)[0]
    return pred, model
    
def forecast_GRU(series, trail_name='GRU', seq_len=30):
    Dtr, Dte, last_seq, scalar = prepare_data_LSTM_GRU(series, seq_len, True)
    model = GRU().to(device)
    model_dir = './saved_model/%s'%trail_name # for early stopping and warm start
    pred, model = train_and_forecast(model, model_dir, Dtr, Dte, last_seq)
    pred = scalar.inverse_transform(pred.reshape(-1,1)).reshape(-1)[0]
    return pred, model




###### 4.3 with TCN

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        """
        其实这就是一个裁剪的模块，裁剪多出来的padding
        """
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        """
        相当于一个Residual block
        :param n_inputs: int, 输入通道数
        :param n_outputs: int, 输出通道数
        :param kernel_size: int, 卷积核尺寸
        :param stride: int, 步长，一般为1
        :param dilation: int, 膨胀系数
        :param padding: int, 填充系数
        :param dropout: float, dropout比率
        """
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        # 经过conv1，输出的size其实是(Batch, input_channel, seq_len + padding)
        self.chomp1 = Chomp1d(padding)  # 裁剪掉多出来的padding部分，维持输出时间步为seq_len
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)  #  裁剪掉多出来的padding部分，维持输出时间步为seq_len
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        """
        :param x: size of (Batch, input_channel, seq_len)
        :return:
        """
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, num_outputs=1, kernel_size=2, dropout=0.2):
        """
        目前paper给出的TCN结构很好的支持每个时刻为一个数的情况, 即sequence结构
        对于每个时刻为一个向量这种一维结构，勉强可以把向量拆成若干该时刻的输入通道，
        对于每个时刻为一个矩阵或更高维图像的情况，就不太好办。

        :param num_inputs: int, 输入通道数
        :param num_channels: list, 每层的hidden_channel数, 例如[25,25,25,25]表示有4个隐层, 每层hidden_channel数为25
        :param kernel_size: int, 卷积核尺寸
        :param dropout: float, drop_out比率
        """
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i   # 膨胀系数：1，2，4，8……
            in_channels = num_inputs if i == 0 else num_channels[i-1]  # 确定每一层的输入通道数
            out_channels = num_channels[i]  # 确定每一层的输出通道数
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]
        self.network = nn.Sequential(*layers)
        self.fc = torch.nn.Linear(num_channels[-1], num_outputs)

    def forward(self, x):
        """
        输入x的结构不同于RNN，一般RNN的size为(Batch, seq_len, channels)或者(seq_len, Batch, channels)，
        这里把seq_len放在channels后面，把所有时间步的数据拼起来，当做Conv1d的输入尺寸，实现卷积跨时间步的操作，
        很巧妙的设计。
        
        :param x: size of (Batch, input_channel, seq_len)
        :return: size of (Batch, output_channel, seq_len)
        """
        x = self.network(x) # [Batch, output_channel, seq_len]
        x = x[:,:,-1] # [Batch, output_channel, 1]
        x = self.fc(x) # [Batch, 1, 1]
        return x


def prepare_data_TCN(series, seq_len, random=True):
    
    # normalize as a column
    scalar = MinMaxScaler()
    scalar.fit(series.reshape(-1,1))
    series = scalar.transform(series.reshape(-1,1)).reshape(-1)
    
    # make dataset
    Dtr = []
    for i in range(len(series)-seq_len):
        x = torch.FloatTensor(np.array(series[i:i+seq_len]).reshape(1,1,seq_len)).to(device) # [batch, input_channel, seq_len]
        y = torch.FloatTensor(np.array(series[i+seq_len]).reshape(1,1)).to(device)
        Dtr.append((x,y))
    last_seq = torch.FloatTensor(np.array(series[-seq_len:]).reshape(1,1,seq_len)).to(device) # only one feature, so add one dimension
    
    # no shuffling (sequences are independent, it's ok to shuffle)
    # Dte, Dtr = MyDataset(Dtr[int(len(Dtr)*0.8):]), MyDataset(Dtr[:int(len(Dtr)*0.8)])
    Dtr = MyDataset(Dtr)
    # shuffle for training
    if random: Dtr, Dte = random_split(Dtr, [int(len(Dtr)*0.8), int(len(Dtr)*0.2)])
    # no shuffle for testing and visualization
    else: Dte = None

    return Dtr, Dte, last_seq, scalar


def forecast_TCN(series, trail_name='TCN', seq_len=30):
    Dtr, Dte, last_seq, scalar = prepare_data_TCN(series, seq_len, True)
    model = TemporalConvNet(num_inputs=1, num_channels=[10,10]).to(device)
    model_dir = './saved_model/%s'%trail_name # for early stopping and warm start
    pred, model = train_and_forecast(model, model_dir, Dtr, Dte, last_seq)
    pred = scalar.inverse_transform(pred.reshape(-1,1)).reshape(-1)[0]
    return pred, model



def prepare_data_TCN_decomp(series, seq_len, random=True):

    series = np.array(series).T # (win_len, n_comp)
    win_len, n_comp = series.shape

    dataY = np.array([ np.sum(series[i+seq_len, :]) for i in range(win_len-seq_len) ]).reshape(win_len-seq_len,1) # a column vector

    # normalize as columns
    scalarX = MinMaxScaler()
    series = scalarX.fit(series).transform(series)
    scalarY = MinMaxScaler()
    dataY = scalarY.fit(dataY).transform(dataY)

    # make dataset
    Dtr = []
    for i in range(win_len-seq_len):
        x = torch.FloatTensor( series[i:i+seq_len, :].T.reshape(1,n_comp,seq_len) ).to(device) # [batch, input_channel, seq_len]
        y = torch.FloatTensor(dataY[i]).reshape(1,1).to(device) # [1,1], in loss: Using a target size (torch.Size([1])) that is different to the input size (torch.Size([1, 1])). This will likely lead to incorrect results due to broadcasting.
        Dtr.append((x,y))
    last_seq = torch.FloatTensor( series[-seq_len:, :].T.reshape(1,n_comp,seq_len) ).to(device) # [batch, input_channel, seq_len]
    
    Dtr = MyDataset(Dtr) 
    # shuffle for training, (sequences are independent, it's ok to shuffle)
    if random: Dtr, Dte = random_split(Dtr, [int(len(Dtr)*0.8), int(len(Dtr)*0.2)])
    # no shuffle for testing and visualization
    else: Dte = None

    return Dtr, Dte, last_seq, scalarY


def forecast_TCN_decomp(series, trail_name='TCN', seq_len=30):
    Dtr, Dte, last_seq, scalar = prepare_data_TCN_decomp(series, seq_len, random=True)
    model = TemporalConvNet(num_inputs=series.shape[0], num_channels=[10,10]).to(device)
    model_dir = './saved_model/%s'%trail_name # for early stopping and warm start
    pred, model = train_and_forecast(model, model_dir, Dtr, Dte, last_seq)
    pred = scalar.inverse_transform(pred.reshape(-1,1)).reshape(-1)[0]
    return pred, model



# visualization helper
def vis_model_performance(model, Dtr, scalar):
    real = []
    pred = []
    model.eval()
    with torch.no_grad():
        for seq, label in Dtr:
            y_pred = model(seq)
            real.extend(label.cpu().detach().numpy())
            pred.extend(y_pred.cpu().detach().numpy())
    pred = scalar.inverse_transform(np.array(pred).reshape(-1,1)).reshape(-1)
    real = scalar.inverse_transform(np.array(real).reshape(-1,1)).reshape(-1)
    rmse = np.sqrt(np.mean( np.square(np.array(real)-np.array(pred)) ))
    plt.figure()
    plt.title('RMSE = %f'%rmse)
    plt.plot(real, label='real')
    plt.plot(pred, label='pred')
    plt.legend()
    plt.show()
    return    



if __name__ == '__main__':

    pass