'''
Predict with hybrid framework (window + restructure + pred).
!!! not tested yet !!!
'''

import pickle
import pandas as pd
import numpy as np

from statsmodels.tsa import stattools
from termcolor import colored
from tqdm import tqdm

from utils import *
from models import *
from series_restr import *
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


# full process
def pred_full(win_len, restr, hi_pred, lo_pred, seq_len=100, pred_len=10, vis=False, val_num=100):

    params = {
        'win_len': win_len, # {800, 500}
        'restr': restr, # {'ssa', 'ssa_ex', 'ceemdan', 'ceemdan_ex', 'eemd', 'emd'}
        'hi_pred': hi_pred, # {'tcn', 'gru', 'lstm', 'bpnn', TODO 'arima'}
        'lo_pred': lo_pred, # {'tcn', 'gru', 'lstm', 'bpnn'}
        'seq_len': seq_len,
        'pred_len': pred_len
    }
    trail_name = 'fu_win%d_%s_%s_%s' %(win_len, restr, hi_pred, lo_pred)
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
    
    restr_dict = {
        'ssa': restr_ssa,           'ssa_ex':restr_ssa_ex, 
        'ceemdan': decomp_ceemdan,  'ceemdan_ex': decomp_ceemdan_ex,
        'eemd': decomp_eemd,        'eemd_ex': decomp_eemd_ex,
        'emd': decomp_emd,          'emd_ex': decomp_emd_ex
    }
    if restr not in restr_dict.keys():
        print('unrecognized restr: ' + restr)
        return


    # load data
    df = pd.read_excel('data\df.xlsx', sheet_name='Sheet1')
    dataY = np.array(df['Cprice'])
    dataX = np.array(df.iloc[:,1:]) # carbon price at the previous time step is also used as feature

    # hold results
    real = np.zeros((val_num, pred_len))
    pred = np.zeros((val_num, pred_len))
    # slide the window, decompose and predict at each step, validate for the last (100) steps
    for ii in tqdm(range(val_num)):
        t = len(dataY) - win_len - pred_len - val_num + ii
        win_x = dataX[t:t+win_len, :] # Xvars, (win_len, n_comp)
        series = dataY[t:t+win_len] # Yvar, (win_len,)
        real[ii,:] = dataY[t+win_len : t+win_len+pred_len] # target, (pred_len,)
        
        # decompose
        restructured = restr_dict[restr](series)
        if restr not in ['ssa_ex', 'ssa']:
            restructured = integr_fuzzen_pwlf(restructured, n_integr=3)
        
        # predict
        # for each subsequence
        for i in range(restructured.shape[0]):
            sub_seq = restructured[i].reshape(-1,1)

            # if white noise, return the mean
            if stattools.q_stat(stattools.acf(sub_seq)[1:11],len(sub_seq))[1][0] > 0.01:
                sub_pred = np.mean(sub_seq)

            # if stationary, goes to high-freq forecast
            elif stattools.adfuller(sub_seq)[1] < 0.01:
                m = model_dict[hi_pred](trail_name+str(i), batch_size)
                try:
                    with HiddenPrints():
                        sub_pred = m.predict(win_x, sub_seq, seq_len, pred_len)
                except Exception as e:
                    sub_pred = sub_seq[-1] # take the last step as prediction
                    print('error in hi_pred: ', hi_pred, e.__class__.__name__, e)
                
            # if non-stationary, goes to low-freq forecast
            else:
                m = model_dict[lo_pred](trail_name+str(i), batch_size)
                try:
                    with HiddenPrints():
                        sub_pred = m.predict(win_x, sub_seq, seq_len, pred_len)
                except Exception as e:
                    sub_pred = sub_seq[-1] # take the last step as prediction
                    print('error in lo_pred: ', lo_pred, e.__class__.__name__, e)
                    
            # integrate the subsequence prediction
            pred[ii,:] += sub_pred.reshape(-1)


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

    pred_full(win_len=500, restr='ceemdan', hi_pred='bpnn', lo_pred='tcn')