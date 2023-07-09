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
    df = pd.read_excel('data\source\CCprice.xlsx', sheet_name='Sheet1')
    Cprice = np.array(df['C_Price'])
    win = Cprice[t:t+win_len]
    dataX = np.array([win]).T
    dataY = win

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


# predict with single framework (windowing + [pred]).
def pred_single(win_len, seq_len, method, pred_len=10, vis=False):

    params = {
        'win_len': win_len, 'seq_len': seq_len, 'method': method
    }
    trail_name = "sg_win%d_seq%d_%s" %(win_len, seq_len, method)
    print(colored(trail_name, 'blue'))

    # method_dict = {
    #     'tcn': TCN_model, 'gru': GRU_model, 'lstm': LSTM_model, 'bpnn': BPNN_model
    # }
    # if method not in method_dict.keys():
    #     print('unrecognized method: ' + method)
    #     return

    # # load data
    # df = pd.read_excel('data\source\CCprice.xlsx', sheet_name='Sheet1')
    # Cprice = np.array(df['C_Price'])

    # # slide the window, predict at each step, validate with the last 100 steps
    # val_num = 100
    # real = np.zeros((val_num, pred_len))
    # pred = np.zeros((val_num, pred_len))
 
    # for i in tqdm(range(val_num)):
    #     t = len(Cprice) - val_num - win_len - pred_len + i
    #     win = Cprice[t : t+win_len]
    #     real[i,:] = Cprice[t+win_len : t+win_len+pred_len]
    #     dataX = np.array([win]).T
    #     dataY = win

    #     m = method_dict[method](trail_name, batch_size)
    #     with HiddenPrints():
    #         p = m.predict(dataX, dataY, seq_len, pred_len)
    #     pred[i,:] = p

    # # store
    # f = open(trail_name+".pkl", "wb")
    # pickle.dump((params, pred, real), f)
    # f.close()

    # load
    f = open("results\\"+trail_name+".pkl", "rb")
    params, pred, real = pickle.load(f)
    f.close()

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
    
    # pred_step(win_len=1000, t=200)

    # win_len = [1000, 500]
    # seq_len = [200, 100]
    # methods = ['tcn', 'gru', 'lstm', 'bpnn']

    pred_single(win_len=500, seq_len=100, method='tcn', vis=False)

    # for i in ['tcn', 'gru', 'lstm', 'bpnn']:
    #     pred_single(win_len=500, seq_len=100, method=i, vis=True)