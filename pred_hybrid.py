'''
Predict with hybrid framework (windowing + restructuring + [pred]).
'''

import matplotlib.pyplot as plt
import numpy as np
import pickle
from statsmodels.tsa import stattools
from tqdm import tqdm

from method import HiddenPrints


# # step1: decompose into subsequences
# # step2: integrate subsequences into hi-freq and lo-freq
# from series_restr import *

# step3: high-freq forecast with ARIMA, in sarimax.py
from sarimax import forecast_arima
  
# step4: low-freq forecast with neural networks, in nn_models.py
from nn_models import forecast_BPNN, forecast_LSTM, forecast_GRU, forecast_TCN


def pred_hybrid(win_len, win_step, restr, hi_pred, lo_pred, vis=False):

    params = {
        'win_len': win_len,
        'win_step': win_step,
        'restr': restr, # {'ceemdan', 'eemd', 'emd', 'ssa'}
        'hi_pred': hi_pred, # {'arima'}
        'lo_pred': lo_pred, # {'bpnn', 'lstm', 'gru', 'tcn'}
    }
    trail_name = 'hb_win%d_sam%d_%s_%s_%s' %(
        params['win_len'], params['win_step'], params['restr'], 
        params['hi_pred'], params['lo_pred'])
    print(trail_name)


    # load windowed and restructured data 
    # load data of win_step=1, params['win_step'] is used to control the simulation progress )
    decomp_datapath = 'data\\restructured\\restr_win%d_sam1_%s.pkl' %(params['win_len'], params['restr'])
    f = open(decomp_datapath, "rb")
    _, win_restrs, win_ys = pickle.load(f)
    f.close()


    # slide the window, and predict at each step
    timesteps = range(0, win_restrs.shape[0], win_step) # sample!!!
    real = np.zeros(len(timesteps))
    pred = np.zeros((win_restrs.shape[1], len(timesteps)))
    
    for t in tqdm(range(len(timesteps))):
        print('\n step: %i' %timesteps[t], end='')
        restr = win_restrs[timesteps[t]]
        win_y = win_ys[timesteps[t]]
        real[t] = win_y

        # predict each subsequence
        for i in range(restr.shape[0]):
            sub_seq = restr[i,:]

            # if stationary, goes to high-freq forecast
            if stattools.adfuller(sub_seq)[1] < 0.001:
                print(' -> high-freq forecasting', end='')
                if params['hi_pred'] == 'arima':
                    try:
                        with HiddenPrints():
                            sub_pred, _ = forecast_arima(sub_seq)
                    except Exception as e:
                        sub_pred = sub_seq[-1] # take the last step as prediction
                        print('error in arima_pred: ',e.__class__.__name__,e)
                else:
                    print('unrecognized hi_pred_method: '+params['hi_pred'])
                    return

            # non-stationary, goes to low-freq forecast
            else:
                print(' -> low-freq forecasting', end='')
                with HiddenPrints():
                    if params['lo_pred'] == 'bpnn':
                        sub_pred, _ = forecast_BPNN(sub_seq, trail_name)
                    elif params['lo_pred'] == 'lstm':
                        sub_pred, _ = forecast_LSTM(sub_seq, trail_name)
                    elif params['lo_pred'] == 'gru':
                        sub_pred, _ = forecast_GRU(sub_seq, trail_name)
                    elif params['lo_pred'] == 'tcn':
                        sub_pred, _ = forecast_TCN(sub_seq, trail_name)
                    else:
                        print('unrecognized lo_pred_method: '+params['lo_pred'])
                        return

            # record
            pred[i,t] = sub_pred

        print(' -> predicted %.3f, observed %.3f' %(np.sum(pred[:,t]), win_y))
        

    # store
    f = open(trail_name+'.pkl', 'wb')
    pickle.dump((params, pred, real), f)
    f.close()

    # # load
    # f = open(trail_name+'.pkl', 'rb')
    # params, pred, real = pickle.load(f)
    # f.close()

    # visualize
    if vis:
        out = np.sum(pred, axis=0)
        rmse = np.sqrt(np.mean( np.square(real-out) ))
        mape = np.mean(np.abs(real-out)/real)*100
        plt.figure()
        plt.title('%s, RMSE=%.2f, MAPE=%.2f%%' %(trail_name,rmse,mape))
        plt.plot(out, label='pred')
        plt.plot(real, label='real')
        plt.legend()
        plt.show()

    return



if __name__ == '__main__':

    # params = {
    #     'win_len': win_len,
    #     'win_step': win_step,
    #     'restr': restr, # {'ceemdan', 'eemd', 'emd', 'ssa'}
    #     'hi_pred': hi_pred, # {'arima'}
    #     'lo_pred': lo_pred, # {'bpnn', 'lstm', 'gru', 'tcn'}
    # }

    pred_hybrid(win_len=200, win_step=1, restr='ssa', hi_pred='arima', lo_pred='tcn', vis=False)