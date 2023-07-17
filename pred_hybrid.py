'''
Predict with hybrid framework (window + restructure + pred).
'''

import pickle
import pandas as pd
import numpy as np

from statsmodels.tsa import stattools
from termcolor import colored
from tqdm import tqdm

from utils import *
from models import *
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


def pred_hybrid(win_len, restr, hi_pred, lo_pred, seq_len=100, pred_len=10, vis=False, val_num=100):

    params = {
        'win_len': win_len, # {800, 500}
        'restr': restr, # {'ssa', 'ssa_ex', 'ceemdan', 'ceemdan_ex', 'eemd', 'emd'}
        'hi_pred': hi_pred, # {'arima'}
        'lo_pred': lo_pred, # {'tcn', 'gru', 'lstm', 'bpnn'}
        'seq_len': seq_len,
        'pred_len': pred_len
    }
    trail_name = 'hb_win%d_%s_%s_%s' %(win_len, restr, hi_pred, lo_pred)
    print(colored(trail_name, 'blue'))


    model_dict = {
        'tcn': TCN_model, 'gru': GRU_model, 'lstm': LSTM_model, 'bpnn': BPNN_model # 'arima': forecast_arima,
    }
    if lo_pred not in model_dict.keys():
        print('unrecognized lo_pred_method: '+lo_pred)
        return
    if hi_pred not in model_dict.keys():
        print('unrecognized hi_pred_method: '+hi_pred)
        return


    # load windowed and restructured price series 
    decomp_datapath = 'data\\restructured\\restr_win%d_%s.pkl' %(win_len, restr)
    f = open(decomp_datapath, "rb")
    _, win_restrs, win_xs, win_ys = pickle.load(f)
    f.close()
    print(win_restrs.shape) # val_num, n_comp, pred_len
    print(win_ys.shape) # val_num, pred_len

    real = np.zeros(win_ys.shape)
    pred = np.zeros(win_ys.shape)
    # slide the window, and predict at each step
    for t in tqdm(range(len(win_restrs))):
        restr = win_restrs[t]
        real[t] = win_ys[t]
        win_x = win_xs[t] # (win_len, n_comp)

        # predict each subsequence
        for i in range(restr.shape[0]):
            sub_seq = restr[i].reshape(-1,1)

            # if white noise, return the mean
            if stattools.q_stat(stattools.acf(sub_seq)[1:11],len(sub_seq))[1][0] > 0.01:
                sub_pred = np.mean(sub_seq)

            # if stationary, goes to high-freq forecast
            elif stattools.adfuller(sub_seq)[1] < 0.01:
                m = model_dict[hi_pred](trail_name+str(i), batch_size)
                with HiddenPrints():
                    sub_pred = m.predict(win_x, sub_seq, seq_len, pred_len)
                # try: TODO
                #     with HiddenPrints():
                #         sub_pred, _ = hi_pred_dict[hi_pred](sub_seq)
                # except Exception as e:
                #     sub_pred = sub_seq[-1] # take the last step as prediction
                #     print('error in hi_pred: ',e.__class__.__name__,e)
                
            # if non-stationary, goes to low-freq forecast
            else:
                m = model_dict[lo_pred](trail_name+str(i), batch_size)
                with HiddenPrints():
                    sub_pred = m.predict(win_x, sub_seq, seq_len, pred_len)
                    
            # integrate the subsequence prediction
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
    show_performance(trail_name, pred, real, vis)

    return

if __name__ == '__main__':
    
    pred_hybrid(win_len=500, restr='ssa_ex', hi_pred='bpnn', lo_pred='tcn', seq_len=100, vis=True)
    pred_hybrid(win_len=500, restr='ssa', hi_pred='bpnn', lo_pred='tcn', seq_len=100, vis=True)
    pred_hybrid(win_len=500, restr='ceemdan_ex', hi_pred='bpnn', lo_pred='tcn', seq_len=100, vis=True)
    pred_hybrid(win_len=500, restr='ceemdan', hi_pred='bpnn', lo_pred='tcn', seq_len=100, vis=True)
    