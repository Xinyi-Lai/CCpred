'''
Predict with single model (windowing + [pred]).
'''

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from termcolor import colored

from utils import *

# models
from sarimax import forecast_arima
from nn_models import forecast_BPNN, forecast_LSTM, forecast_GRU, forecast_TCN


def pred_single(win_len, win_step, method, vis=False):

    params = {
        'win_len': win_len,
        'win_step': win_step,
        'method': method # {'arima', 'bpnn', 'lstm', 'gru', 'tcn'}
    }
    trail_name = "sg_win%d_sam%d_%s" %(params['win_len'], params['win_step'], params['method'])
    print(colored(trail_name, 'blue'))

    # load data
    df = pd.read_excel('data\source\CCprice.xlsx', sheet_name='Sheet1')
    Cprice = np.array(df['C_Price'])


    # slide the window, and predict at each step
    timesteps = range(0, len(Cprice)-win_len-1, win_step) # sample!!!
    real = np.zeros(len(timesteps))
    pred = np.zeros(len(timesteps))
 
    for t in tqdm(range(len(timesteps))): # sample!!!
        win_x = Cprice[timesteps[t] : timesteps[t]+win_len]
        win_y = Cprice[timesteps[t]+win_len]
        
        with HiddenPrints():
            if method == 'arima':
                step_pred, _ = forecast_arima(win_x)
            elif method == 'bpnn':
                step_pred, _ = forecast_BPNN(win_x, trail_name)
            elif method == 'lstm':
                step_pred, _ = forecast_LSTM(win_x, trail_name)
            elif method == 'gru':
                step_pred, _ = forecast_GRU(win_x, trail_name)
            elif method == 'tcn':
                step_pred, _ = forecast_TCN(win_x, trail_name)
            else:
                print('unrecognized pred_method: ' + method)
                break

        real[t] = win_y
        pred[t] = step_pred

    # store
    f = open(trail_name+".pkl", "wb")
    pickle.dump((params, pred, real), f)
    f.close()

    # # load
    # f = open(trail_name+".pkl", "rb")
    # params, pred, real = pickle.load(f)
    # f.close()

    rmse = cal_rmse(real, pred)
    mape = cal_mape(real, pred)
    print('%s, RMSE=%.2f, MAPE=%.2f%%' %(trail_name, rmse, mape))

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
    
    # pred_methods = ['tcn', 'arima', 'gru', 'lstm', 'bpnn']
    # for i in pred_methods:
    #     pred_single(win_len=200, win_step=1, method=i, vis=False)
    
    pred_single(win_len=200, win_step=1, method='tcn', vis=False)