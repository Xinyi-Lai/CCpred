'''
For the report, no windowing, predict with single and hybrid framework
'''
import openpyxl
import pandas as pd
import pickle
import pmdarima as pm
import numpy as np
from copy import deepcopy
from termcolor import colored
from tqdm import tqdm

from forecasting_methods import *
from nn_models import *
from series_restr import series_restr_func
from utils import *

### global parameters
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
        try:
            _, not_stationary = pm.arima.ADFTest().should_diff(subY)
        except Exception as e:
            print(e)
            not_stationary = False
        
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
def CCpred_nowin(WhichMarket, PredMethod, vis, save):
    if len(PredMethod.split('-')) == 1:
        SingleOrHybrid, RestrMethod, PredMethods = 'sg', None, [PredMethod]
    else:
        SingleOrHybrid, RestrMethod, PredMethods = 'hb', PredMethod.split('-')[0], PredMethod.split('-')[1:]
    trail_name = WhichMarket + '-' + PredMethod
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
        pred, real = pred_nowin_single(data, trail_name, PredMethods[0].lower())
    elif SingleOrHybrid == 'hb':
        pred, real = pred_nowin_hybrid(data, trail_name, RestrMethod.lower(), PredMethods[0].lower(), PredMethods[1].lower())
    else:
        print('unrecognized framework: ' + SingleOrHybrid)
        return

    show_performance(trail_name, pred, real, vis)

    if save:
        res_path = 'res/results_'+WhichMarket+'.xlsx'
        writer = pd.ExcelWriter(res_path, mode='a', engine='openpyxl', if_sheet_exists='replace')
        pd.DataFrame(pred).to_excel(writer, sheet_name=trail_name+'-pred')
        pd.DataFrame(real).to_excel(writer, sheet_name=trail_name+'-real')
        writer.close()

    return


def post_processing(WhichMarket):
    res_path = 'res/results_'+WhichMarket+'.xlsx'
    # store excel data into result dict
    result = dict([ ('step'+str(i), {'real':None}) for i in range(10)])
    wb = openpyxl.load_workbook(res_path)
    for sheet in wb.worksheets:
        if sheet.title == 'Sheet1':
            continue
        # read excel data into table
        table = []
        for row in sheet.rows:
            row_list = []
            for cell in row:
                row_list.append(cell.value)
            table.append(row_list)
        table = np.array(table)
        table = table[1:,1:]
        # reorganize and store table into result dict
        for i in range(10):
            title_split = sheet.title.split('-')
            if title_split[-1] == 'real':
                title = 'real'
            else:
                title = '-'.join(list(sheet.title.split('-')[1:-1]))
            result['step'+str(i)][title] = table[:,i]

    # print(result.keys()) # dict_keys(['step0', 'step1', 'step2', 'step3', 'step4', 'step5', 'step6', 'step7', 'step8', 'step9'])
    # print(result['step0'].keys()) # dict_keys(['real', 'ARIMA', 'SSA-SVR-ARIMA', 'SVR', 'SSA-ARIMA-SVR', 'CEEMDAN-SVR-ARIMA', 'VMD-SVR-ARIMA'])
    
    # reorganize result dict and calculate metrics
    methods = ['SSA-SVR-ARIMA', 'CEEMDAN-SVR-ARIMA', 'VMD-SVR-ARIMA', 'SSA-ARIMA-SVR', 'ARIMA', 'SVR']
    metrics = dict([ (m, []) for m in methods])
    for m in methods:
        for step in result.keys():
            metrics[m].append(cal_rmse(result[step]['real'], result[step][m]))
            metrics[m].append(cal_mape(result[step]['real'], result[step][m]))

    # store metrics into excel
    writer = pd.ExcelWriter('res/res_'+WhichMarket+'.xlsx', engine='openpyxl', mode='a', if_sheet_exists='replace')
    pd.DataFrame(metrics).to_excel(writer, sheet_name='metrics')
    for i in result.keys():
        pd.DataFrame(result[i]).to_excel(writer, sheet_name=i)
    writer.close()

    return



if __name__ == '__main__':

    CCpred_nowin('GZ', 'SSA-SVR-ARIMA', vis=True, save=False)
    

    # WhichMarket = 'EU' # ['GZ', 'HB', 'SH', 'BJ', 'FJ', 'CQ', 'TJ', 'SZ', 'EU']
    # vis, save = False, True
    # CCpred_nowin(WhichMarket, 'ARIMA', vis, save)
    # CCpred_nowin(WhichMarket, 'SVR', vis, save)
    # CCpred_nowin(WhichMarket, 'SSA-SVR-ARIMA', vis, save)
    # CCpred_nowin(WhichMarket, 'SSA-ARIMA-SVR', vis, save)
    # CCpred_nowin(WhichMarket, 'CEEMDAN-SVR-ARIMA', vis, save)
    # CCpred_nowin(WhichMarket, 'VMD-SVR-ARIMA', vis, save)
    # post_processing(WhichMarket)
    
    # import timeit
    # start = timeit.default_timer()
    # CCpred_nowin('EU', 'sg', '', ['gru'], vis=False, save=False)
    # end = timeit.default_timer()
    # print('Time: ', end-start)
    
    # pred_win_single_arima('arima', True)
    # pred_win_hybrid_arima('ssa', 'arima', 'tcn', True)



    
    