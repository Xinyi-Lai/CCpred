""" a super class for neural network forecasting models 
    common methods are implemented here
"""

import matplotlib.pyplot as plt 
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split

from utils import *
from models.early_stopping import EarlyStopping


class NN_model(object):
    """ Provides a common interface to all neural network forecasting model classes.
        Stores common functions, such as train and forecast.
    """

    def __init__(self, model_name, batch_size) -> None:
        # assign in __init__()
        self.model_name = model_name    # for early stopping and warm start
        self.batch_size = batch_size    # support batching
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Use gpu if available
        # assign in prepare_data()
        self.n_in = None                # number of input features (subsequences and variables)
        self.n_out = None               # number of output features (prediction steps)
        self.scalar = None              # output scalar (y scalar)
        self.dataset = None             # the whole reorganized dataset
        self.last_seq = None            # the last input sequence for prediction
        # assign in init_model()
        self.model = None               # the model
        # assign in get_forecast()
        self.pred = None                # forecast from last_seq
        return


    def prepare_data(self):
        """ transform data into batched tensors, reorganize to adapt to different model structures, store dataset
            implement in subclass, otherwise raises: NotImplementedError
        """
        raise NotImplementedError()


    def init_model(self):
        """ initialize nn model instance, super params are specified for each model
            implement in subclass, otherwise raises: NotImplementedError
        """
        raise NotImplementedError()
    

    def train_model(self):
        """ all nn models share the same training process: (except for seq2seq)
            1. split the dataset into training set and validation set
            2. load pretrained model and define loss, optimizer, early-stopping, etc.
            3. iterate through epochs, training and validating
        """
        # # split training and testing sets without shuffling
        # # Dte, Dtr = MyDataset(Dtr[int(len(Dtr)*0.8):]), MyDataset(Dtr[:int(len(Dtr)*0.8)])
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
                y_pred = self.model(seq)
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
                    y_pred = self.model(seq)
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
        """ all nn models share the same forecasting process: (except for seq2seq)
            1. get forecast results from last seq
            2. reshape and scale
        """
        self.pred = self.model(self.last_seq).cpu().detach().numpy() # (1,n_out)
        self.pred = self.scalar.inverse_transform(self.pred)
        return


    def predict(self, dataX, dataY, seq_len, pred_len):
        """ all nn models share the same prediction process:
        """
        self.prepare_data(dataX, dataY, seq_len, pred_len)
        self.init_model()
        self.train_model()
        self.get_forecast()
        return self.pred


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
                ypred = self.model(seq)
                real.extend(label.cpu().numpy())
                pred.extend(ypred.cpu().numpy())
        pred = self.scalar.inverse_transform(np.array(pred))
        real = self.scalar.inverse_transform(np.array(real))

        show_performance(self.model_name, pred, real, plot)
        return pred, real


    def load_model(self, model_name):
        model_path = os.path.join('saved_model', model_name, 'best_model.pth')
        try:
            if os.path.exists(model_path):
                self.model.load_state_dict(torch.load(model_path))
        except Exception as e:
            print('error in loading model: ',e.__class__.__name__,e)

        print('model %s loaded from file' %model_name)
        return



class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __getitem__(self, item):
        return self.data[item]
    def __len__(self):
        return len(self.data)