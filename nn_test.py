'''
test refactored nn models with single prediction.
TODO support arima??
'''

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from termcolor import colored

from utils import *
from models import TCN_model, LSTM_model, GRU_model, BPNN_model


import random
import torch

# TODO Fix the random seed to ensure the reproducibility of the experiment
random_seed = 10
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)


batch_size = 1


def pred_step(win_len, t):

    df = pd.read_excel('data\source\CCprice.xlsx', sheet_name='Sheet1')
    Cprice = np.array(df['C_Price'])
    win = Cprice[t:t+win_len]
    dataX = np.array([win]).T
    dataY = win

    batch_size = 1

    m1 = TCN_model('TCN-step', batch_size)
    m2 = GRU_model('GRU-step', batch_size)
    m3 = LSTM_model('LSTM-step', batch_size)
    m4 = BPNN_model('BPNN-step', batch_size)

    for m in [m1,m2,m3,m4]:
        # with HiddenPrints():
        pred = m.predict(dataX, dataY, seq_len=200, pred_len=10)
        print(pred)
        m.vis_performance(True)

    return


def pred_single(win_len, seq_len, method, pred_len=10, vis=False):

    params = {
        'win_len': win_len,
        'seq_len': seq_len,
        'method': method # {'tcn', 'gru', 'lstm', 'bpnn', 'arima'}
    }
    trail_name = "sg_win%d_seq%d_%s" %(win_len, seq_len, method)
    print(colored(trail_name, 'blue'))

    # load data
    df = pd.read_excel('data\source\CCprice.xlsx', sheet_name='Sheet1')
    Cprice = np.array(df['C_Price'])

    # slide the window, predict at each step, validate with the last 100 steps
    val_num = 100
    real = np.zeros((val_num, pred_len))
    pred = np.zeros((val_num, pred_len))
 
    for i in tqdm(range(val_num)):
        t = len(Cprice) - val_num - win_len - pred_len + i
        win = Cprice[t : t+win_len]
        real[i,:] = Cprice[t+win_len : t+win_len+pred_len]
        dataX = np.array([win]).T
        dataY = win

        # if method == 'arima':
        #     p = forecast_ARIMA(win_x, trail_name)
        if method == 'tcn':
            m = TCN_model(trail_name, batch_size)
        elif method == 'gru':
            m = GRU_model(trail_name, batch_size)
        elif method == 'lstm':
            m = LSTM_model(trail_name, batch_size)
        elif method == 'bpnn':
            m = BPNN_model(trail_name, batch_size)
        else:
            print('unrecognized pred_method: ' + method)
            break
        with HiddenPrints():
            p = m.predict(dataX, dataY, seq_len, pred_len)
        pred[i,:] = p

    # store
    f = open(trail_name+".pkl", "wb")
    pickle.dump((params, pred, real), f)
    f.close()

    # load
    # f = open(trail_name+".pkl", "rb")
    # params, pred, real = pickle.load(f)
    # f.close()


    # performance
    print('performance of %s' %trail_name)
    for i in range(pred.shape[1]):
        rmse = cal_rmse(real[:,i], pred[:,i])
        mape = cal_mape(real[:,i], pred[:,i])
        print('the %ith column, RMSE=%.2f, MAPE=%.2f%%' %(i, rmse, mape))
    if vis:
        # plot the first, middle, and last columns
        f, axes = plt.subplots(1,3)
        f.suptitle('performance of %s' %trail_name)
        for idx, icol in enumerate([0, pred.shape[1]//2, -1]):
            ax = axes[idx]
            r = real[:,icol]
            p = pred[:,icol]
            ax.plot(r, label='real')
            ax.plot(p, label='pred')
            ax.set_title('col%d, RMSE=%.2f, MAPE=%.2f%%' %(icol, cal_rmse(r,p), cal_mape(r,p)))
            ax.legend()
        plt.show()

    return



if __name__ == '__main__':
    
    # pred_methods = ['tcn', 'arima', 'gru', 'lstm', 'bpnn']
    # for i in pred_methods:
    #     pred_single(win_len=1000, seq_len=200, method=i, vis=False)

    # pred_step(win_len=1000, t=200)

    pred_single(win_len=1000, seq_len=200, method='gru', vis=False)

