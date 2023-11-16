'''
For the report, no windowing, predict with single and hybrid framework
'''

import pandas as pd
import numpy as np
import pmdarima as pm
from copy import deepcopy
from termcolor import colored
from tqdm import tqdm
import pickle

from forecasting_methods import *
from nn_models import *
from series_restr import series_restr_func
from utils import *

# global parameters
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
            city: one of ['Guangzhou', 'Hubei', 'Shanghai', 'Beijing', 'Fujian', 'Chongqing', 'Tianjin', 'Shenzhen']
        Returns:
            data (np.array), shape (n_samples, 1+n_features): the whole dataset, target ($city) at the first column, followed by other cities, followed by other features 
    '''
    df = pd.read_excel('data/df_chn.xlsx', sheet_name='df-chn-itpl') # FIXME, do intl after dropna?
    cities = ['Guangzhou', 'Hubei', 'Shanghai', 'Beijing', 'Fujian', 'Chongqing', 'Tianjin', 'Shenzhen']
    other_cities = list(set(cities) - set([city]))
    cols = [city] + other_cities + ['EU-CC', 'WTI-Oil', 'Brent-Oil', 'Zhengzhou-Coal', 'Dalian-Coal', 'Rtd-Coal', 'US-NatGas', 'SH-FOil', 'US-FOil', 'CSI300', 'US-DJI', 'USD-CNY']
    data = np.array(df[cols])
    # print(data.shape) # (1324,20)
    return data



### predict with a given method (no windowing)
def series_pred_func(data, method, trail_name=None, batch_size=batch_size, seq_len=seq_len, pred_len=pred_len, train_ratio=train_ratio):
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
        # m.print_summary()
        pred, real = m.test_model(test_ratio=1-train_ratio)

    elif method == 'arima': 
        pred, real = pred_arima(data, seq_len, pred_len, train_ratio)

    elif method == 'svr':
        ml_x, ml_y = ml_prepare_data(data, seq_len, pred_len)
        pred, real = pred_svr(ml_x, ml_y, train_ratio)

    else:
        print('unrecognized method: ' + method)
        return

    return pred, real


# no windowing, single framework, predict
def pred_nowin_single(data, trail_name, method):
    return series_pred_func(data, method, trail_name)


# no windowing, hybrid framework, restructure + predict + sum
def pred_nowin_hybrid(data, trail_name, restr, hi_pred, lo_pred):

    dataX = data[:,1:]
    dataY = data[:,0].reshape(-1,1)
    reconstr = series_restr_func(dataY, decomp_method=restr, n_integr=3)

    valid_len = len(dataY)-seq_len-pred_len+1 # real n_samples for training and testing, the first seq_len is in training set, the last pred_len is used in target
    split = int( valid_len * train_ratio ) + seq_len # the first seq is in training set
    test_size = len(dataY) - split - pred_len + 1
    pred = np.zeros((test_size, pred_len))
    real = np.zeros((test_size, pred_len))
    # NOTE: summing reconstr may not give the original series (VMD or rounding errors)
    # construct real from dataY
    for i in range(test_size):
        real[i,:] = dataY[split+i: split+i+pred_len, :].reshape(-1)

    for i in range(reconstr.shape[0]):
        subY = reconstr[i,:].reshape(-1,1)
        _, not_stationary = pm.arima.ADFTest().should_diff(subY)
        subData = np.concatenate((subY, dataX), axis=1)

        # if stationary, goes to high-freq forecast
        if ~not_stationary:
            print('subseq'+str(i)+': high-freq forecast with '+hi_pred+'...')
            sub_pred, sub_real = series_pred_func(subData, hi_pred, trail_name+str(i), batch_size, seq_len, pred_len, train_ratio)

        # if non-stationary, goes to low-freq forecast
        else:
            print('subseq'+str(i)+': low-freq forecast with '+lo_pred+'...')
            sub_pred, sub_real = series_pred_func(subData, lo_pred, trail_name+str(i), batch_size, seq_len, pred_len, train_ratio)
        pred += sub_pred

    return pred, real



### the whole workflow
def CCpred_nowin(WhichMarket, SingleOrHybrid, RestrMethod, PredMethods, vis, save):
    trail_name = ('%s-%s-%s-' %(WhichMarket, SingleOrHybrid, RestrMethod)) + '-'.join(PredMethods)
    print(colored(trail_name, 'green'))

    markets = {'EU':'EU-ETS', 'GZ':'Guangzhou', 'HB':'Hubei', 'SH':'Shanghai', 'BJ':'Beijing', 'FJ':'Fujian', 'CQ':'Chongqing', 'TJ':'Tianjin', 'SZ':'Shenzhen'}
    if markets[WhichMarket] == 'EU-ETS':
        data = loaddata_eu()
    elif WhichMarket in markets.keys():
        data = loaddata_chn(markets[WhichMarket])
    else:
        print('unrecognized market: ' + WhichMarket)
        return

    if SingleOrHybrid == 'sg':
        pred, real = pred_nowin_single(data, trail_name, PredMethods[0])
    elif SingleOrHybrid == 'hb':
        pred, real = pred_nowin_hybrid(data, trail_name, RestrMethod, PredMethods[0], PredMethods[1])
    else:
        print('unrecognized framework: ' + SingleOrHybrid)
        return

    show_performance(trail_name, pred, real, vis)

    if save:
        writer = pd.ExcelWriter('res/result_report_'+WhichMarket+'.xlsx', mode='a', engine='openpyxl')
        pd.DataFrame(pred).to_excel(writer, sheet_name=trail_name+'-pred')
        pd.DataFrame(real).to_excel(writer, sheet_name=trail_name+'-real')
        writer.save()
        writer.close()

    return



if __name__ == '__main__':

    # CCpred_nowin('EU', 'hb', 'vmd', ['arima','svr'], vis=False, save=True)
    # CCpred_nowin('EU', 'hb', 'vmd', ['svr','arima'], vis=False, save=True)
    # CCpred_nowin('EU', 'hb', 'vmd', ['arima','tcn'], vis=False, save=True)
    # CCpred_nowin('EU', 'hb', 'vmd', ['tcn','arima'], vis=False, save=True)
    # CCpred_nowin('EU', 'hb', 'vmd', ['svr','tcn'], vis=False, save=True)
    # CCpred_nowin('EU', 'hb', 'vmd', ['tcn','svr'], vis=False, save=True)

    CCpred_nowin('SZ', 'sg', '', ['arima'], vis=False, save=True)
    CCpred_nowin('SZ', 'sg', '', ['svr'], vis=False, save=True)
    CCpred_nowin('SZ', 'hb', 'ssa', ['svr', 'arima'], vis=False, save=True)
    CCpred_nowin('SZ', 'hb', 'ssa', ['arima', 'svr'], vis=False, save=True)
    CCpred_nowin('SZ', 'hb', 'ceemdan', ['svr', 'arima'], vis=False, save=True)
    CCpred_nowin('SZ', 'hb', 'vmd', ['svr', 'arima'], vis=False, save=True)
    
    # TJ, SZ

    # data = loaddata_eu()
    # data = loaddata_chn('Hubei')

    # pred_nowin_single('tcn', data, vis=True)
    # pred_nowin_single('gru', data, vis=True)
    # pred_nowin_single('lstm', data, vis=True)
    # pred_nowin_single('mlp', data, vis=True)

    # pred, real = pred_nowin_single(data, 'arima', vis=False)
    # pred, real = pred_nowin_single(data, 'svr', vis=False, save=True)
    # pred, real = pred_nowin_hybrid(data, 'ssa', 'arima', 'svr', vis=False)
    # pred, real = pred_nowin_hybrid(data, 'ssa', 'svr', 'arima', vis=False)
    # pred, real = pred_nowin_hybrid(data, 'ssa', 'arima', 'arima', vis=False)

    # pred_win_single_arima('arima', True)
    # pred_win_hybrid_arima('ssa', 'arima', 'tcn', True)



    
    