'''
For the report, no windowing, predict with single and hybrid framework
'''

import pandas as pd
import numpy as np
import pmdarima as pm
from copy import deepcopy
from termcolor import colored
from tqdm import tqdm

from models import *
from series_restr import series_restr_func
from utils import *

# parameters
batch_size = 1
pred_len = 10
seq_len = 100
train_ratio = 0.9

### load data - EU-ETS
df = pd.read_excel('data/df_eu.xlsx', sheet_name='df-eu')
Xcols = ['Cprice', 'Eprice', 'BrentOil', 'CrudeOilF', 'TTF-NatGas', 'NatGasF', 'Coal', 'GasolineF', 'DJI', 'S&P500', 'USD-EUR']
# note: Cprice at previous steps are also features for the current step
dataX = np.array(df[Xcols]) # (1305, 11)
dataY = np.array(df['Cprice']).reshape(-1,1) # (1305, 1)


### load data - CHN-ETS
# df = pd.


def series_pred_func(dataX, dataY, method, trail_name=None, batch_size=1, seq_len=100, pred_len=10, train_ratio=0.9):
    ''' predict a series with a given method
        Args:
            dataX (np.array): features, shape (n_samples, n_features)
            dataY (np.array): targets, shape (n_samples, 1)
            method: 'tcn', 'lstm', 'gru', 'bpnn', 'arima'
            trail_name, batch_size, seq_len, pred_len, train_ratio
        Returns:
            pred (np.array): predictions, shape (n_test_samples, pred_len)
            real (np.array): real values, shape (n_test_samples, pred_len)
            n_test_samples = int( (1-train_ratio) * (n_samples-seq_len_pred_len+1) )
    '''

    method_dict = { 'tcn': TCN_model, 'gru': GRU_model, 'lstm': LSTM_model, 'bpnn': BPNN_model }
    if method in method_dict.keys():
        m = method_dict[method](trail_name, batch_size)
        m.prepare_data(dataX, dataY, seq_len, pred_len)
        m.init_model()
        m.train_model(train_ratio=train_ratio, val_ratio=0.2)
        # m.load_model()
        # m.print_summary()
        pred, real = m.test_model(test_ratio=1-train_ratio)

    elif method == 'arima': # note: actually still windowing
        split = int( (len(dataY)-seq_len-pred_len+1) * train_ratio )  + seq_len # the first seq is in training set
        trainY, testY = dataY[:split], dataY[split:]

        # fit model with training set
        model = pm.auto_arima(  
            trainY, start_p=1, start_q=1,
            max_p=10, max_q=10,         # maximum p and q
            information_criterion='aic',# 'aic', 'aicc', 'bic', 'hqic', 'oob'
            d=None,                     # let model determine 'd'
            test='adf',                 # use adftest to find optimal 'd', or 'kpss', 'pp'
            m=1,                        # frequency of series, m=1 means non-seasonal
            seasonal=False,             # no Seasonality
            trace=False,                # don't print status on the fits
            error_action='ignore',      # don't want to know if an order does not work
            suppress_warnings=True,     # don't want convergence warnings
            stepwise=True               # The stepwise algorithm can be significantly faster than
                                        # a non-stepwise selection (i.e., essentially a grid search) 
                                        # and is less likely to over-fit the model. 
        ) 
        print(model.summary())

        # predict with test set
        pred = np.zeros((len(testY)-pred_len+1, pred_len))
        real = np.zeros((len(testY)-pred_len+1, pred_len))
        for i in tqdm(range(len(testY)-pred_len+1)):
            model_copy = deepcopy(model)
            if i != 0: model_copy.update(testY[:i], maxiter=0)
            new_pred = model_copy.predict(n_periods=pred_len, return_conf_int=False)
            pred[i,:] = new_pred
            real[i,:] = dataY[split+i: split+i+pred_len].reshape(-1)

    else:
        print('unrecognized method: ' + method)
        return

    return pred, real


### no windowing, single framework, predict with single nn-model
def pred_nowin_single(method, vis=True):
    trail_name = "sg_nowin_%s" %(method)
    print(colored(trail_name, 'blue'))
    pred, real = series_pred_func(dataX, dataY, method, trail_name, batch_size, seq_len, pred_len, train_ratio)
    show_performance(trail_name, pred, real, vis)
    return


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

    # pred_nowin_single('tcn', vis=True)
    # pred_nowin_single('gru', vis=True)
    # pred_nowin_single('lstm', vis=True)
    # pred_nowin_single('bpnn', vis=True)
    pred_nowin_single('arima', vis=True)

    # pred_nowin_hybrid('ssa', 'arima', 'tcn', vis=True)

    
    