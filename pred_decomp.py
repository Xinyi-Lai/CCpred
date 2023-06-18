'''
Predict with decomposed data (windowing + restructuring + [multi-feature pred]).
'''

import matplotlib.pyplot as plt
import numpy as np
import pickle
from tqdm import tqdm
from termcolor import colored

from utils import *

from nn_models import forecast_TCN_decomp


def pred_decomp(win_len, win_step, restr, method, vis=False):

    params = {
        'win_len': win_len,
        'win_step': win_step,
        'restr': restr, # {'ceemdan', 'eemd', 'emd', 'ssa'}
        'method': method, # {'bpnn', 'lstm', 'gru', 'tcn'}
    }
    trail_name = 'dc_win%d_sam%d_%s_%s' %(params['win_len'], params['win_step'], params['restr'], params['method'])
    print(colored(trail_name, 'blue'))


    # load windowed and restructured data 
    # load data of win_step=1, params['win_step'] is used to control the simulation progress )
    decomp_datapath = 'data\\restructured\\restr_win%d_sam1_%s.pkl' %(params['win_len'], params['restr'])
    f = open(decomp_datapath, "rb")
    _, win_restrs, win_ys = pickle.load(f)
    f.close()
    
    # only using the last 1000 samples for simulation FIXME
    # win_restrs = win_restrs[-1000:]
    # win_ys = win_ys[-1000:]
    print(win_restrs.shape) # (1000,5,200) (1000,n_comp,win_len)

    # slide the window, and predict at each step
    timesteps = range(0, win_restrs.shape[0], win_step) # sample!
    real = np.zeros(len(timesteps))
    pred = np.zeros(len(timesteps))
    
    for t in tqdm(range(len(timesteps))):
        # print('\n step: %i' %timesteps[t], end='')
        real[t] = win_ys[timesteps[t]]
        restr = win_restrs[timesteps[t]]

        if t == 0:
            # take all subsequences as input features
            with HiddenPrints():
                if params['method'] == 'tcn':
                    pred[t], _ = forecast_TCN_decomp(restr, trail_name)
                elif params['method'] == 'gru':
                    pred[t], _ = forecast_GRU_decomp(restr, trail_name)
                # elif params['method'] == 'bpnn':
                #     sub_pred, _ = forecast_BPNN(sub_seq, trail_name)
                # elif params['method'] == 'lstm':
                #     sub_pred, _ = forecast_LSTM(sub_seq, trail_name)
                else:
                    print('unrecognized method: '+params['method'])
                    return
        else:
            pred[t] = real[t-1]

        # print('step %i -> predicted %.3f, observed %.3f' %(timesteps[t], pred[t], real[t]))
        

    # store
    f = open(trail_name+'.pkl', 'wb')
    pickle.dump((params, pred, real), f)
    f.close()

    # # load
    # f = open(trail_name+'.pkl', 'rb')
    # params, pred, real = pickle.load(f)
    # f.close()

    rmse = cal_rmse(real, pred)
    mape = cal_mape(real, pred)
    print(colored('%s, RMSE=%.2f, MAPE=%.2f%%' %(trail_name, rmse, mape), 'green'))

    # visualize
    if vis:
        plt.figure()
        plt.title('%s, RMSE=%.2f, MAPE=%.2f%%' %(trail_name, rmse, mape))
        plt.plot(pred, label='pred')
        plt.plot(real, label='real')
        plt.legend()
        plt.show()

    return



if __name__ == '__main__':

    # params = {
    #     'win_len': win_len,
    #     'win_step': win_step,
    #     'restr': restr, # {'ssa', 'ssa_ex', 'ceemdan', 'ceemdan_ex', 'eemd', 'emd', }
    #     'method': method, # {'bpnn', 'lstm', 'gru', 'tcn'}
    # }
    
    pred_decomp(win_len=200, win_step=1, restr='ssa', method='tcn', vis=True)