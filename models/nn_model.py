""" a super class for neural network forecasting models 
    common methods are implemented here
"""

import matplotlib.pyplot as plt 
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchinfo import summary

from utils import *
from models.early_stopping import EarlyStopping


class NN_model(object):
    """ Provides a common interface to all neural network forecasting model classes.
        Stores common functions, such as train and forecast.
    """

    def __init__(self, model_name, batch_size, lr) -> None:
        # assign in __init__()
        self.model_name = model_name    # for early stopping and warm start
        self.batch_size = batch_size    # support batching
        self.lr = lr                    # learning rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Use gpu if available
        # assign in prepare_data()
        self.in_n = None                # number of input features (subsequences and variables)
        self.out_n = None               # number of output features (prediction steps)
        self.in_dim = None              # shape of input, for model summary
        self.scalar = None              # output scalar (y scalar)
        self.dataset = None             # the whole reorganized dataset
        self.last_seq = None            # the last input sequence for rolling prediction
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
    

    def train_model(self, train_ratio=0.9, val_ratio=0.2):
        """ all nn models share the same training process: (except for seq2seq)
            0. split out training set (for nowin pred, train_ratio=0.8, for rolling pred with windowing, train_ratio=1)
            1. load pretrained model
            2. define loss, optimizer, early-stopping, etc.
            3. iterate through epochs, training and validating
            4. stop and save
        """

        ### 1. split dataset for training and validation
        Dtr = self.dataset[ : int(len(self.dataset)*train_ratio)]
        # sequences are independent, so it's ok to shuffle
        split = int(len(Dtr)*(1-val_ratio))
        Dtr, Dva = random_split(Dtr, [split, len(Dtr)-split])
        # wrap with DataLoader
        Dtr = DataLoader(MyDataset(Dtr), batch_size=self.batch_size)
        Dva = DataLoader(MyDataset(Dva), batch_size=self.batch_size)

        ### 2. warm start if applicable
        model_path = os.path.join('saved_model', self.model_name, 'best_model.pth')
        try:
            if os.path.exists(model_path):
                self.model.load_state_dict(torch.load(model_path))
        except Exception as e:
            print('error in warm start: ',e.__class__.__name__,e)
        
        ### 3. loss, optimizer, early-stopping
        # define loss and optimizer
        loss_func = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        # initialize an early-stopping instance
        model_dir = os.path.join('saved_model', self.model_name)
        early_stopping = EarlyStopping(model_dir)
        
        ### 4. training and validating
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
                for seq, label in Dva:
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


    def get_forecast(self): # FIXME
        """ all nn models share the same forecasting process: (except for seq2seq)
            1. get forecast results from last seq
            2. reshape and scale
        """
        self.pred = self.model(self.last_seq).cpu().detach().numpy() # (1,n_out)
        self.pred = self.scalar.inverse_transform(self.pred)
        return


    def predict(self, dataX, dataY, seq_len, pred_len): # FIXME
        """ all nn models share the same prediction process: (for rolling prediction)
        """
        self.prepare_data(dataX, dataY, seq_len, pred_len)
        self.split_data(test_ratio=0, val_ratio=0.2) # rolling prediction
        self.init_model()
        self.train_model()
        self.get_forecast()
        return self.pred


    def test_model(self, test_ratio=0.1):
        """ test the trained/loaded model on the testing set
            return the predicted and real values (inverse scaled)
        """
        split = int(len(self.dataset)*(1-test_ratio))
        Dte = DataLoader(MyDataset(self.dataset[split:]), batch_size=self.batch_size)

        real = []
        pred = []
        loss_func = torch.nn.MSELoss()
        test_loss = []
        self.model.eval()
        with torch.no_grad(): # no backward propagation, no computational graph
            for seq, label in Dte:
                ypred = self.model(seq)
                test_loss.append(loss_func(ypred, label).item())
                real.extend(label.cpu().numpy())
                pred.extend(ypred.cpu().numpy())
        
        test_loss = np.mean(test_loss)
        print('test_loss {:.8f}'.format(test_loss))

        pred = self.scalar.inverse_transform(np.array(pred))
        real = self.scalar.inverse_transform(np.array(real))
        return pred, real


    def load_model(self):
        model_path = os.path.join('saved_model', self.model_name, 'best_model.pth')
        try:
            self.model.load_state_dict(torch.load(model_path))
            print('model %s loaded from file' %self.model_name)
        except Exception as e:
            print('error in loading model: ',e.__class__.__name__,e)


    def print_summary(self):
        summary(self.model, self.in_dim)
        return



class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __getitem__(self, item):
        return self.data[item]
    def __len__(self):
        return len(self.data)