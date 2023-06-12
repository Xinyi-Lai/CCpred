import os
from tqdm import tqdm
from termcolor import colored
import numpy as np
import matplotlib.pyplot as plt
import optuna
import torch
from torch.optim.lr_scheduler import StepLR

from early_stopping import EarlyStopping
from model import *
from utils import *
from PrepareData import *

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # I only have cpu

params_CNN_LSTM = {
    # data params
    'batch_size':   8, # trial.suggest_categorical('batch_size', [2,4,8,16,32,64]), # 
    'seq_len':      30, # trial.suggest_int('seq_len', 5,60), # 
    'prev_step':    30, # trial.suggest_int('prev_step', 5,60), # 
    # model params
    'cnn_out_channels': 7, # trial.suggest_int('cnn_out_channels', 2,8), # 
    'cnn_kernel_size':  4, # trial.suggest_int('cnn_kernel_size', 2,4), # 
    'lstm_hidden_size': 7, # trial.suggest_int('lstm_hidden_size', 2,8), # 
    'lstm_num_layers':  1, # trial.suggest_int('lstm_num_layers', 1,4), # 
    'input_size': None, # calculated
    # learning params
    'loss_func':    'MSELoss', # trial.suggest_categorical('loss_func', list(loss_funcs.keys())), # 
    'optimizer':    'RMSprop', # trial.suggest_categorical('optimizer', ['Adam', 'RMSprop', 'SGD']), # 
    'lr':           0.0028046, # trial.suggest_float('lr', 1e-5, 1e-1, log=True), # 
    'weight_decay': 0.000376358, # trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True), # 
}


data_path = './testdata/CCprice.xlsx'
model_dir = './saved_CNN_LSTM'

loss_funcs = {
    "MSELoss": torch.nn.MSELoss(),
    "L1Loss": torch.nn.L1Loss(),
    "SmoothL1Loss": torch.nn.SmoothL1Loss(),
}


def test_and_show_results(model, Dte, output=False, plot_start=0, plot_end=-1):
    print(colored('-------------Testing-------------', 'blue'))

    real = []
    pred = []
    model.eval() # testing mode
    with torch.no_grad(): # no backward propagation, no computational graph
        for (seq, label) in Dte:
            y_pred = model(seq)
            real.extend(label.detach().numpy())
            pred.extend(y_pred.detach().numpy())
    real = real[plot_start : plot_end]
    pred = pred[plot_start : plot_end]
    mse = MSE_value(real, pred)
    mae = MAE_value(real, pred)

    if output:
        # print('load data from file: ' + data_path)
        # print('seq_len = %d, prev_step = %d, batch_size = %d ' %(seq_len, prev_step, batch_size))
        print(model)
        print('Mean Squared Error: %.10f; Mean Absolute Error: %.10f' %(mse, mae))
        date_col = pd.read_excel(data_path, sheet_name='Sheet1')
        date_col = pd.to_datetime(date_col['Date'])
        date_col = date_col[plot_start : plot_end]
        plt.figure()
        plt.plot(date_col, real, 'r', label='real')
        plt.plot(date_col, pred, 'b', label='pred')
        plt.xlabel('Time')
        plt.ylabel('Normalized Carbon price')
        plt.legend(loc='best')
        plt.title('Carbon price prediction, MSE={:.6f}, MAE={:.3f}'.format(mse, mae))
        plt.show()
    
    return mse



def train_model(params, trial=None, output=False):

    ######### prepare data for training and testing #########
    batch_size = params['batch_size']
    seq_len = params['seq_len']
    prev_step = params['prev_step']
    Dtr, val, Dte, input_feature_num = load_split_data(data_path, train_ratio=0.8, val_ratio=0.1)
    Dtr = process(Dtr, seq_len, prev_step, batch_size, True)
    val = process(val, seq_len, prev_step, batch_size, True)
    Dte = process(Dte, seq_len, prev_step, batch_size, False)
    params['input_size'] = input_feature_num + prev_step

    ######### prepare model for training and testing #########
    # initialize early stopping instance and defining model saved path
    early_stopping = EarlyStopping(model_dir, patience=20)
    # initialize a network instance
    model = CNN_LSTM(params)
    print(model)
    # define model's loss function, optimizer, and scheduler
    loss_func = loss_funcs[params['loss_func']]
    optimizer = getattr(torch.optim, params['optimizer'])(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
    
    ################## not in optuna ##################
    if not trial:
        # warm start if applicable
        model_path = os.path.join(model_dir, 'best_model.pth')
        if os.path.exists(model_path):
            print(colored('------------loading model for warm start-------------', 'green'))
            model.load_state_dict(torch.load(model_path))
            print(colored('-----------------model load complete-----------------', 'green'))
    ################## not in optuna ##################


    ######################### training #########################

    epochs = 500
    print(colored('-------------Training-------------', 'blue'))
    for e in range(epochs):
        print(colored('-------------Epoch_{}-------------'.format(e+1), 'green'))
        
        # train the model with the training set
        train_loss = []
        model.train() # training mode
        for (seq, label) in Dtr: # tqdm(Dtr)
            # forward
            y_pred = model(seq)
            loss = loss_func(y_pred, label)
            train_loss.append(loss.item())
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        train_loss = np.mean(train_loss)
        
        # validate the model with the validation set
        val_loss = []
        model.eval() # testing mode
        with torch.no_grad(): # no backward propagation, no computational graph
            for (seq, label) in val:
                y_pred = model(seq)
                l = loss_func(y_pred, label)
                val_loss.append(l.item())
        val_loss = np.mean(val_loss)
        
        print('epoch {:03d} train_loss {:.8f} val_loss {:.8f}'.format(e+1, train_loss, val_loss))
        
        ################## for optuna only ##################
        if trial:
            trial.report(val_loss, e)
            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        ################## for optuna only ##################

        # if there is no improvement for some epochs, 'early_stop' is set to True
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break


    ######################### testing #########################
    
    # if not in optuna, show the whole curve
    if not trial:
        _, _, Dte, input_feature_num = load_split_data(data_path, train_ratio=0, val_ratio=0)
        Dte = process(Dte, seq_len, prev_step, batch_size, False)
    # else report the loss of test set

    return test_and_show_results(model, Dte, output=output)


if __name__ == "__main__":
    
    test_mse = train_model(params_CNN_LSTM, output=True)
