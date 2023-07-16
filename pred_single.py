'''
Predict with single framework (windowing + [pred]).
FUTURE: support arima??
'''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import random
from termcolor import colored
import torch
from tqdm import tqdm

from utils import *
from models import TCN_model, LSTM_model, GRU_model, BPNN_model


# TODO Fix the random seed to ensure the reproducibility of the experiment
random_seed = 10
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)


batch_size = 1


# test refactored nn models
def pred_step(win_len, t):
    df = pd.read_excel('data\df.xlsx', sheet_name='Sheet1')
    dataY = np.array(df['Cprice'])
    dataX = np.array(df.iloc[:,1:]) # carbon price at the previous time step is also used as feature

    dataY = dataY[t:t+win_len, np.newaxis] # (win_len, 1)
    dataX = dataX[t:t+win_len, :] # (win_len, n_comp)

    m1 = TCN_model('step-TCN', batch_size)
    m2 = GRU_model('step-GRU', batch_size)
    m3 = LSTM_model('step-LSTM', batch_size)
    m4 = BPNN_model('step-BPNN', batch_size)
    for m in [m1,m2,m3,m4]:
        # with HiddenPrints():
        pred = m.predict(dataX, dataY, seq_len=100, pred_len=10)
        print(pred)
        m.vis_performance(True)

    return


# predict with single framework (windowing + [pred]).
def pred_single(win_len, seq_len, method, pred_len=10, vis=False, val_num=100):

    params = { 'win_len': win_len, 'seq_len': seq_len, 'method': method }
    trail_name = "sg_win%d_seq%d_%s" %(win_len, seq_len, method)
    print(colored(trail_name, 'blue'))

    method_dict = { 'tcn': TCN_model, 'gru': GRU_model, 'lstm': LSTM_model, 'bpnn': BPNN_model }
    if method not in method_dict.keys():
        print('unrecognized method: ' + method)
        return

    # load data
    df = pd.read_excel('data\df.xlsx', sheet_name='Sheet1')
    dataY = np.array(df['Cprice'])
    dataX = np.array(df.iloc[:,1:]) # carbon price at the previous time step is also used as feature

    # slide the window, predict at each step, validate with the last 100 steps
    real = np.zeros((val_num, pred_len))
    pred = np.zeros((val_num, pred_len))
 
    for i in tqdm(range(val_num)):
        t = len(dataY) - win_len - pred_len - val_num + i
        real[i,:] = dataY[t+win_len : t+win_len+pred_len]
        winY = dataY[t:t+win_len, np.newaxis] # (win_len, 1)
        winX = dataX[t:t+win_len, :] # (win_len, n_comp)
        m = method_dict[method](trail_name, batch_size)
        with HiddenPrints():
            p = m.predict(winX, winY, seq_len, pred_len)
        pred[i,:] = p

    # store
    f = open(trail_name+".pkl", "wb")
    pickle.dump((params, pred, real), f)
    f.close()

    # # load
    # f = open("results\\"+trail_name+".pkl", "rb")
    # params, pred, real = pickle.load(f)
    # f.close()

    # performance
    print('performance of %s' %trail_name)
    for i in range(pred.shape[1]):
        rmse = cal_rmse(real[:,i], pred[:,i])
        mape = cal_mape(real[:,i], pred[:,i])
        print('col %d: RMSE=%.2f, MAPE=%.2f%%' %(i, rmse, mape))
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
    
    # pred_step(win_len=800, t=100)

    # win_len = [1000, 500]
    # seq_len = [200, 100]
    # methods = ['tcn', 'gru', 'lstm', 'bpnn']

    pred_single(win_len=800, seq_len=100, method='gru', vis=True, val_num=20)

    # for i in ['tcn', 'gru', 'lstm', 'bpnn']:
    #     pred_single(win_len=500, seq_len=100, method=i, vis=True)