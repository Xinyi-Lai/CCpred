'''
Predict with hybrid framework (windowing + restructuring + [pred]).
'''

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa import stattools
from tqdm import tqdm
from termcolor import colored

from utils import *
from models import TCN_model, LSTM_model, GRU_model, BPNN_model
from sarimax import forecast_arima

import random
import torch

# TODO Fix the random seed to ensure the reproducibility of the experiment
random_seed = 10
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)


batch_size = 1


def pred_hybrid(win_len, restr, hi_pred, lo_pred, seq_len=200, pred_len=10, vis=False):

    params = {
        'win_len': win_len, # {1000, 500}
        'restr': restr, # {'ceemdan', 'eemd', 'emd', 'ssa'}
        'hi_pred': hi_pred, # {'arima'}
        'lo_pred': lo_pred, # {'tcn', 'gru', 'lstm', 'bpnn'}
        'seq_len': seq_len,
        'pred_len': pred_len
    }
    trail_name = 'hb_win%d_%s_%s_%s' %(win_len, restr, hi_pred, lo_pred)
    print(colored(trail_name, 'blue'))

    hi_pred_dict = {
        'tcn': TCN_model, 'bpnn': BPNN_model, # 'arima': forecast_arima,
    }
    if hi_pred not in hi_pred_dict.keys():
        print('unrecognized hi_pred_method: '+hi_pred)
        return

    lo_pred_dict = {
        'tcn': TCN_model, 'gru': GRU_model, 'lstm': LSTM_model, 'bpnn': BPNN_model
    }
    if lo_pred not in lo_pred_dict.keys():
        print('unrecognized lo_pred_method: '+lo_pred)
        return

    # load windowed and restructured data 
    decomp_datapath = 'data\\restructured\\restr_win%d_%s.pkl' %(win_len, restr)
    f = open(decomp_datapath, "rb")
    _, win_restrs, win_ys = pickle.load(f)
    f.close()
    print(win_restrs.shape) # len(win_restrs), n_comp, pred_len
    print(win_ys.shape) # len(win_restrs), pred_len


    # # load data TODO in pred_full
    # df = pd.read_excel('data\source\CCprice.xlsx', sheet_name='Sheet1')
    # Cprice = np.array(df['C_Price'])

    real = np.zeros(win_ys.shape)
    pred = np.zeros(win_ys.shape)
    # slide the window, and predict at each step
    for t in tqdm(range(len(win_restrs))):
        restr = win_restrs[t]
        real[t] = win_ys[t]

        # predict each subsequence
        for i in range(restr.shape[0]):
            sub_seq = restr[i]
            dataX = np.array([sub_seq]).T
            dataY = sub_seq

            # if white noise, return the mean
            if stattools.q_stat(stattools.acf(sub_seq)[1:11],len(sub_seq))[1][0] > 0.01:
                sub_pred = np.mean(sub_seq)

            # if stationary, goes to high-freq forecast
            elif stattools.adfuller(sub_seq)[1] < 0.01:
                # print(' -> high-freq forecasting', end='')
                m = hi_pred_dict[hi_pred](trail_name+str(i), batch_size)
                with HiddenPrints():
                    sub_pred = m.predict(dataX, dataY, seq_len, pred_len)
                # try: TODO
                #     with HiddenPrints():
                #         sub_pred, _ = hi_pred_dict[hi_pred](sub_seq)
                # except Exception as e:
                #     sub_pred = sub_seq[-1] # take the last step as prediction
                #     print('error in hi_pred: ',e.__class__.__name__,e)
                
            # if non-stationary, goes to low-freq forecast
            else:
                # print(' -> low-freq forecasting', end='')
                m = lo_pred_dict[lo_pred](trail_name+str(i), batch_size)
                with HiddenPrints():
                    sub_pred = m.predict(dataX, dataY, seq_len, pred_len)
                    
            # record
            pred[t,:] += sub_pred.reshape(-1)


    # store
    f = open(trail_name+".pkl", "wb")
    pickle.dump((params, pred, real), f)
    f.close()

    # # load
    # f = open("results\\"+trail_name+".pkl", "rb")
    # _, pred, real = pickle.load(f)
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
            ax.set_title('col%d: RMSE=%.2f, MAPE=%.2f%%' %(icol, cal_rmse(r,p), cal_mape(r,p)))
            ax.legend()
        plt.show()

    return

if __name__ == '__main__':
    
    pred_hybrid(win_len=1000, restr='ssa_ex', hi_pred='bpnn', lo_pred='gru', seq_len=200)
    
    pred_hybrid(win_len=1000, restr='ssa', hi_pred='bpnn', lo_pred='gru', seq_len=200)

    pred_hybrid(win_len=1000, restr='ceemdan_ex', hi_pred='bpnn', lo_pred='gru', seq_len=200)
    
    pred_hybrid(win_len=1000, restr='ceemdan', hi_pred='bpnn', lo_pred='gru', seq_len=200)
    