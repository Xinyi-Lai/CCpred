'''
For the report, no windowing, predict with single and hybrid framework
'''

import pandas as pd
import numpy as np
import pmdarima as pm
from copy import deepcopy
from termcolor import colored
from tqdm import tqdm

from forecasting_methods import *
from nn_models import *
from series_restr import series_restr_func
from utils import *

# parameters
batch_size = 1
pred_len = 10
seq_len = 100
train_ratio = 0.9


########## load data from files ##########

def loaddata_eu():
    ''' load data - EU-ETS
        Returns:
            data (np.array), shape (n_samples, 1+n_features): the whole dataset, target ('Cprice') at the first column, followed by other features 
    '''
    df = pd.read_excel('data/df_eu.xlsx', sheet_name='df-eu')
    cols = ['Cprice', 'Eprice', 'BrentOil', 'CrudeOilF', 'TTF-NatGas', 'NatGasF', 'Coal', 'GasolineF', 'DJI', 'S&P500', 'USD-EUR']
    data = np.array(df[cols])
    # print(data.shape) # (1305, 11)
    return data

def loaddata_chn(city):
    ''' load data - CHN-ETS
        Args:
            city: one of ['Guangzhou', 'Hubei', 'Shanghai', 'Beijing', 'Fujian', 'Chongqing']
        Returns:
            data (np.array), shape (n_samples, 1+n_features): the whole dataset, target ($city) at the first column, followed by other cities, followed by other features 
    '''
    df = pd.read_excel('data/df_chn.xlsx', sheet_name='df-chn-itpl') # FIXME, do intl after dropna?
    cities = ['Guangzhou', 'Hubei', 'Shanghai', 'Beijing', 'Fujian', 'Chongqing']
    other_cities = list(set(cities) - set([city]))
    cols = [city] + other_cities + ['EU-CC', 'WTI-Oil', 'Brent-Oil', 'Zhengzhou-Coal', 'Dalian-Coal', 'Rtd-Coal', 'US-NatGas', 'SH-FOil', 'US-FOil', 'CSI300', 'US-DJI', 'USD-CNY']
    data = np.array(df[cols])
    # print(data.shape) # (1226,18)
    return data



### predict with a given method (no windowing)
def series_pred_func(data, method, trail_name=None, batch_size=1, seq_len=100, pred_len=10, train_ratio=0.9):
    ''' predict a series with a given method
        Args:
            data (np.array), shape (n_samples, 1+n_features): target at the first column, followed by other features 
            method: 'tcn', 'gru', 'lstm', 'mlp', 'arima', 'svr'
            trail_name, batch_size, seq_len, pred_len, train_ratio
        Returns:
            pred (np.array): predictions, shape (n_test_samples, pred_len)
            real (np.array): real values, shape (n_test_samples, pred_len)
            n_test_samples = int( (1-train_ratio) * (n_samples-seq_len_pred_len+1) )
    '''

    method_dict = { 'tcn': TCN_model, 'gru': GRU_model, 'lstm': LSTM_model, 'mlp': MLP_model }
    if method in method_dict.keys():
        # note: Cprice at previous steps are also features for the current step
        dataX = data[:,:]
        dataY = data[:,0].reshape(-1,1)
        m = method_dict[method](trail_name, batch_size)
        m.prepare_data(dataX, dataY, seq_len, pred_len)
        m.init_model()
        m.train_model(train_ratio=train_ratio, val_ratio=0.2)
        # m.load_model()
        m.print_summary()
        pred, real = m.test_model(test_ratio=1-train_ratio)

    elif method == 'arima': 
        pred, real = pred_arima(data, seq_len, pred_len, train_ratio)

    elif method == 'svr':
        ml_x, ml_y = ml_prepare_data(data, seq_len, pred_len)
        pred, real = pred_svr(ml_x, ml_y, train_ratio, auto=False)

    else:
        print('unrecognized method: ' + method)
        return

    return pred, real


### no windowing, single framework, predict with single nn-model
def pred_nowin_single(method, data, vis=True):
    trail_name = "sg_nowin_%s" %(method)
    print(colored(trail_name, 'blue'))
    pred, real = series_pred_func(data, method, trail_name, batch_size, seq_len, pred_len, train_ratio)
    show_performance(trail_name, pred, real, vis)
    return pred, real


# no windowing, hybrid framework, restructure + predict + sum
def pred_nowin_hybrid(restr, hi_pred, lo_pred, vis=True):
    trail_name = 'hb_%s_%s_%s' %(restr, hi_pred, lo_pred)
    print(colored(trail_name, 'blue'))

    pred = []
    real = []
    reconstr = series_restr_func(dataY, decomp_method=restr, n_integr=3)

    for i in range(reconstr.shape[0]):
        subY = reconstr[i,:].reshape(-1,1)
        _, not_stationary = pm.arima.ADFTest().should_diff(subY)

        # if stationary, goes to high-freq forecast
        if ~not_stationary:
            print('subseq'+str(i)+': high-freq forecast with '+hi_pred+'...')
            sub_pred, sub_real = series_pred_func(dataX, subY, hi_pred, trail_name+str(i), batch_size, seq_len, pred_len, train_ratio)

        # if non-stationary, goes to low-freq forecast
        else:
            print('subseq'+str(i)+': low-freq forecast with '+lo_pred+'...')
            sub_pred, sub_real = series_pred_func(dataX, subY, lo_pred, trail_name+str(i), batch_size, seq_len, pred_len, train_ratio)
        
        pred.append(sub_pred)
        real.append(sub_real)

    pred = np.sum(np.array(pred), axis=0)
    real = np.sum(np.array(real), axis=0)
    
    show_performance(trail_name, pred, real, vis=True)

    return




if __name__ == '__main__':

    data = loaddata_eu()
    # data = loaddata_chn('Guangzhou')

    # pred_nowin_single('tcn', data, vis=True)
    # pred_nowin_single('gru', data, vis=True)
    # pred_nowin_single('lstm', data, vis=True)
    pred_nowin_single('mlp', data, vis=True)

    # pred_nowin_single('arima', data, vis=True)
    # pred_nowin_single('svr', data, vis=True)

    # pred_nowin_hybrid('ssa', 'arima', 'arima', vis=True)
    # pred_nowin_hybrid('ssa', 'arima', 'svr', vis=True)
    # pred_nowin_hybrid('ssa', 'arima', 'tcn', vis=True)

    # pred_win_single_arima('arima', True)
    # pred_win_hybrid_arima('ssa', 'arima', 'tcn', True)

    
    