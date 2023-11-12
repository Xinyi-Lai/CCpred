import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tqdm import tqdm
from termcolor import colored
import torch
from torch.utils.data import DataLoader, Dataset
    

def load_split_data(path, train_ratio=0.8, val_ratio=0.1):
    """ load data from excel with pandas, split data into training, validaton, and test sets
        Args:
            path (str): file path
            train_ratio (float, optional): ratio of training set. Defaults to 0.8. Normally Dtr:val:Dte = 8:1:1 or 6:2:2
            val_ratio (float, optional): ratio of validation set. Defaults to 0.1. Normally Dtr:val:Dte = 8:1:1 or 6:2:2
        Returns:
            Dtr (tuple): training set   (feature as np.array, label as np.array)
            val (tuple): validation set (feature as np.array, label as np.array)
            Dte (tuple): test set       (feature as np.array, label as np.array)
            input_feature_num (int): number of columns taken as features
    """
    
    # load from excel
    print(colored('------------loading data from {}-------------'.format(path), 'blue'))
    dataset = pd.read_excel(path, sheet_name='Sheet1')
    print('Data loaded, # of records: %d, # of columns: %d' %(dataset.shape[0], dataset.shape[1]))
    featureX = ['E_Price']
    featureY = ['C_Price']
    dataX = dataset[featureX]
    dataY = dataset[featureY]
    input_feature_num = len(featureX)
    
    # normalization
    scalerX = MinMaxScaler() # StandardScaler() # 
    scalerX.fit(dataX)
    dataX = scalerX.transform(dataX)
    scalerY = MinMaxScaler() # StandardScaler() # 
    scalerY.fit(dataY)
    dataY = scalerY.transform(dataY)
    # scalerY.inverse_transform(pred)
    # dataX = np.array(dataX)
    # dataY = np.array(dataY)

    # split
    split1 = int(dataX.shape[0] * train_ratio)
    split2 = int(dataX.shape[0] * (train_ratio + val_ratio))
    trainX = dataX[0 : split1]
    trainY = dataY[0 : split1]
    valX = dataX[split1 : split2]
    valY = dataY[split1 : split2]
    testX = dataX[split2 : ]
    testY = dataY[split2 : ]
    
    return (trainX, trainY), (valX, valY), (testX, testY), input_feature_num



class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data
 
    def __getitem__(self, item):
        return self.data[item]
 
    def __len__(self):
        return len(self.data)

def process(data, seq_len, prev_step, batch_size, shuffle, msg=None):
    """ process dataset with pytorch's Dataset and DataLoader
        Args:
            data (tuple): dataset to be processed (feature as np.array, label as np.array)
            seq_len (int): seq_len or time_step for lstm network, period of data (96)
            prev_step (int): number of previous labels adding to feature (24)
            batch_size (int): batch size (32)
            shuffle (bool): whether each batch is suffled (can be False in training and validation, because the time-sequence info is considered in seq_len)
            msg (str, optional): msg to print. Defaults to None.
        Returns:
            seq (pytorch's DataLoader): 
                for x,y in seq we have:
                    x.shape = torch.Size([batch_size, seq_len, input_feature_num+prev_step])
                    y.shape = torch.Size([batch_size, 1])
    """
    if msg:
        print('Processing %s ...' %msg)
    dataX, dataY = data
    seq = []
        
    # 单步预测, 将最近 prev_step 的 label 加入 feature, 再叠前序 seq_len 行作为 lstm 网络的 time_step
    # input: [batch_size, seq_len, num_features+prev_step]
    # x.append([dataX[i+j]]) # Creating a tensor from a list of numpy.ndarrays is extremely slow. Please consider converting the list to a single numpy.ndarray with numpy.array() before converting to a tensor
    for i in range(seq_len+prev_step-1, len(dataX)):
    # for i in tqdm(range(seq_len+prev_step-1, len(dataX))): # progress bar
        y = torch.FloatTensor(dataY[i]) # 当前行的输出
        x = [] 
        for j in range(seq_len): 
            xj = np.append(dataX[i-j], dataY[i-j-prev_step : i-j]) # 将最近 prev_step 个 label 加入 feature
            x = np.append(xj, x) # 往前叠上前序的 seq_len 行
        x = torch.FloatTensor(x)
        x = x.reshape(seq_len, -1)
        seq.append((x,y))
    
    seq = MyDataset(seq)
    seq = DataLoader(dataset=seq, batch_size=batch_size, shuffle=shuffle, num_workers=0, drop_last=True)
    return seq    



if __name__ == "__main__":
    
    path = './testdata/CCprice.xlsx'

    # example
    seq_len = 30
    prev_step = 10
    batch_size = 16
    
    Dtr, val, Dte, input_feature_num = load_split_data(path, train_ratio=0.8, val_ratio=0.1)
    Dtr = process(Dtr, seq_len, prev_step, batch_size, True, 'training set')
    val = process(val, seq_len, prev_step, batch_size, True, 'validation set')
    Dte = process(Dte, seq_len, prev_step, batch_size, False, 'test set')

    input_size = input_feature_num + prev_step

    print("len(Dtr)="+str(len(Dtr)))
    for x,y in Dtr:
        print("y.shape="+str(y.shape)) # y.shape=torch.Size([16, 1])
        print("x.shape="+str(x.shape)) # x.shape=torch.Size([16, 30, 11])
        break
    for x,y in Dte:
        print(x)
        print(y)
