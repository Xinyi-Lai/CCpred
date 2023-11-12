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
def loaddata_eu():
    df = pd.read_excel('data/df_eu.xlsx', sheet_name='df-eu')
    # note: Cprice at previous steps are also features for the current step
    Xcols = ['Cprice', 'Eprice', 'BrentOil', 'CrudeOilF', 'TTF-NatGas', 'NatGasF', 'Coal', 'GasolineF', 'DJI', 'S&P500', 'USD-EUR']
    dataX = np.array(df[Xcols]) # (1305, 11)
    dataY = np.array(df['Cprice']).reshape(-1,1) # (1305, 1)
    # print(dataX.shape, dataY.shape)
    return dataX, dataY


def loaddata_chn(city):
    ### load data - CHN-ETS
    df = pd.read_excel('data/df_chn.xlsx', sheet_name='df-chn-itpl') # FIXME, do intl after dropna?
    df.dropna(subset=[city], inplace=True)
    # note: Cprice at previous steps are also features for the current step
    Xcols = [city, 'EU-CC', 'WTI-Oil', 'Brent-Oil', 'Zhengzhou-Coal', 'Dalian-Coal', 'Rtd-Coal', 'US-NatGas', 'SH-FOil', 'US-FOil', 'CSI300', 'US-DJI', 'USD-CNY']
    dataX = np.array(df[Xcols]) # GZ, HB, BJ: (1306, 13); SH: 1292, FJ, CQ
    dataY = np.array(df[city]).reshape(-1,1) # GZ, HB, BJ: (1306, 1)
    print(dataX.shape, dataY.shape)
    return dataX, dataY



### SVR support vector regression
def pred_svr(dataX, dataY, seq_len=100, pred_len=10, train_ratio=0.9, auto=False):
    from sklearn.svm import SVR
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import RandomizedSearchCV, cross_val_score

    # reorganize data for ml models (align features and targets)
    ml_x, ml_y = [], []
    for i in range(len(dataY)-seq_len-pred_len+1):
        xi = np.concatenate((dataX[i+seq_len-1,:].reshape(-1), dataY[i:i+seq_len-1].reshape(-1)) ) # -1 是因为当前的Cprice也是feature
        yi = dataY[i+seq_len : i+seq_len+pred_len].reshape(-1)
        ml_x.append(xi)
        ml_y.append(yi)
    ml_x, ml_y = np.array(ml_x), np.array(ml_y)
    # print(ml_x.shape, ml_y.shape)

    split = int(len(ml_y)*train_ratio)
    real = np.zeros((len(ml_y)-split, pred_len))
    pred = np.zeros((len(ml_y)-split, pred_len))

    # predict each y in ml_y
    for i in tqdm(range(pred_len)):
        x, y = ml_x, ml_y[:,i]

        scalarX, scalarY = StandardScaler(), StandardScaler()
        x = scalarX.fit_transform(x)
        y = scalarY.fit_transform(y.reshape(-1,1)).reshape(-1)
        x_tran, x_test = x[:split], x[split:]
        y_tran, y_test = y[:split], y[split:]

        clf = SVR( kernel='linear', C=0.1 ) # baseline
        
        if auto: # auto-tune parameters, not always better
            baseline_score = np.mean(cross_val_score(clf, x_tran, y_tran, cv=5, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1))
            param_grid = {
                'kernel': ['rbf', 'linear', 'poly', 'sigmoid'], 
                'C': [pow(10,i) for i in range(-2,3)],
                'gamma': ['scale','auto'] + [pow(10,i) for i in range(-5,-1)], 
            }
            random_search = RandomizedSearchCV(SVR(), param_grid, cv=5, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)
            random_search.fit(x_tran, y_tran)
            if random_search.best_score_ > baseline_score:
                best_params = random_search.best_estimator_.get_params()
                print('\nSVR: best_score_ %.6f > baseline_score %.6f' %(random_search.best_score_, baseline_score))
                print(random_search.best_estimator_)
                clf = SVR(
                    kernel = best_params['kernel'], # 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'
                    C = best_params['C'],           # penalty parameter C of the error term, default 1.0
                    gamma = best_params['gamma'],   # kernel coefficient for 'rbf', 'poly' and 'sigmoid', scale = 1 / (n_features*x.var()) auto = 1 / n_features
                    # epsilon = 0.1,      # epsilon in the epsilon-SVR model,
                    # tol = 0.001,        # tolerance for stopping criterion
                    # degree = 3,         # degree of polynomial, ignored with other kernels
                    # coef0 = 0.0,        # independent term in kernel function, important in 'poly' and 'sigmoid'
                    # shrinking = True,   # whether to use the shrinking heuristic
                    # cache_size = 200,   # kernel cache size in MB
                    # verbose = False,    # enable verbose output
                    # max_iter = -1,      # limit on iterations within solver, -1 for no limit
                )

        clf.fit(x_tran, y_tran)
        y_pred = clf.predict(x_test)
        y_pred = scalarY.inverse_transform(y_pred.reshape(-1,1))
        y_real = scalarY.inverse_transform(y_test.reshape(-1,1))

        real[:,i] = y_real.reshape(-1)
        pred[:,i] = y_pred.reshape(-1)

    return pred, real


### predict with a given method (no windowing)
def series_pred_func(dataX, dataY, method, trail_name=None, batch_size=1, seq_len=100, pred_len=10, train_ratio=0.9):
    ''' predict a series with a given method
        Args:
            dataX (np.array): features, shape (n_samples, n_features)
            dataY (np.array): targets, shape (n_samples, 1)
            method: 'tcn', 'lstm', 'gru', 'mlp', 'arima', 'svr'
            trail_name, batch_size, seq_len, pred_len, train_ratio
        Returns:
            pred (np.array): predictions, shape (n_test_samples, pred_len)
            real (np.array): real values, shape (n_test_samples, pred_len)
            n_test_samples = int( (1-train_ratio) * (n_samples-seq_len_pred_len+1) )
    '''

    method_dict = { 'tcn': TCN_model, 'gru': GRU_model, 'lstm': LSTM_model, 'mlp': MLP_model }
    if method in method_dict.keys():
        m = method_dict[method](trail_name, batch_size)
        m.prepare_data(dataX, dataY, seq_len, pred_len)
        m.init_model()
        m.train_model(train_ratio=train_ratio, val_ratio=0.2)
        # m.load_model()
        m.print_summary()
        pred, real = m.test_model(test_ratio=1-train_ratio)

    elif method == 'arima': # note: actually still windowing
        split = int( (len(dataY)-seq_len-pred_len+1) * train_ratio )  + seq_len # the first seq is in training set
        print(split)
        trainY, testY = dataY[:split], dataY[split:]

        # fit model with training set
        model = pm.auto_arima(  
            trainY, start_p=1, start_q=1,
            max_p=3, max_q=3,         # maximum p and q
            information_criterion='aic',# 'aic', 'aicc', 'bic', 'hqic', 'oob'
            d=0,                        # d=None let model determine 'd' # FIXME
            test='adf',                 # use adftest to find optimal 'd', or 'kpss', 'pp'
            m=1,                        # frequency of series, m=1 means non-seasonal
            seasonal=False,             # no Seasonality
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

    elif method == 'svr':
        pred, real = pred_svr(dataX, dataY, seq_len, pred_len, train_ratio, auto=False)

    else:
        print('unrecognized method: ' + method)
        return

    return pred, real


### no windowing, single framework, predict with single nn-model
def pred_nowin_single(method, dataX, dataY, vis=True):
    trail_name = "sg_nowin_%s" %(method)
    print(colored(trail_name, 'blue'))
    pred, real = series_pred_func(dataX, dataY, method, trail_name, batch_size, seq_len, pred_len, train_ratio)
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



### predict with a given method (with windowing)
def series_win_pred_func(winX, winY, method, trail_name=None, batch_size=1, seq_len=100, pred_len=10, train_ratio=0.9):
    ''' predict a series with a given method FIXME:
        Args:
            dataX (np.array): features, shape (n_samples, n_features)
            dataY (np.array): targets, shape (n_samples, 1)
            method: 'tcn', 'lstm', 'gru', 'mlp', 'arima'
            trail_name, batch_size, seq_len, pred_len, train_ratio
        Returns:
            pred (np.array): predictions, shape (n_test_samples, pred_len)
            real (np.array): real values, shape (n_test_samples, pred_len)
            n_test_samples = int( (1-train_ratio) * (n_samples-seq_len_pred_len+1) )
    '''

    method_dict = { 'tcn': TCN_model, 'gru': GRU_model, 'lstm': LSTM_model, 'mlp': MLP_model }
    if method in method_dict.keys():
        m = method_dict[method](trail_name, batch_size)
        m.prepare_data(dataX, dataY, seq_len, pred_len)
        m.init_model()
        m.train_model(train_ratio=train_ratio, val_ratio=0.2)
        # m.load_model()
        m.print_summary()
        pred, real = m.test_model(test_ratio=1-train_ratio)

    elif method == 'arima': 
        # fit model 
        model = pm.auto_arima(  
            winY, start_p=1, start_q=1,
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
        # print(model.summary())
        pred = model.predict(n_periods=pred_len, return_conf_int=False)

    else:
        print('unrecognized method: ' + method)
        return

    return pred





def sliding_pred_arima(winY, model):

    if model == None:
        # fit model with winY
        model = pm.auto_arima(  
            winY, start_p=1, start_q=1,
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
        # print(model.summary())
        # make prediction
        pred = model.predict(n_periods=pred_len, return_conf_int=False)

    else:
        # otherwise, update with new observed value (the last observation of the sliding window)
        model.update(winY[-1])
        # make prediction
        pred = model.predict(n_periods=pred_len, return_conf_int=False)

    return pred, model


def sliding_pred_nn(winX, winY, method, model_name):
    model_dict = { 'tcn': TCN_model, 'gru': GRU_model, 'lstm': LSTM_model, 'mlp': MLP_model }
    model = model_dict[method](model_name, batch_size)
    pred = model.predict(winX, winY, seq_len, pred_len)
    return pred


def pred_win_single_arima(method, vis):
    trail_name = "sg_win_%s" %(method)
    print(colored(trail_name, 'blue'))

    val_num = 100
    win_len = 800

    # slide the window, predict at each step, validate with the last (100) steps
    real = np.zeros((val_num, pred_len))
    pred = np.zeros((val_num, pred_len))
    m = None
 
    for i in tqdm(range(val_num)):
        t = len(dataY) - win_len - pred_len - val_num + i
        real[i,:] = dataY[t+win_len : t+win_len+pred_len, :].reshape(-1)
        winY = dataY[t:t+win_len, :] # (win_len, 1)
        winX = dataX[t:t+win_len, :] # (win_len, n_comp)
        # p = series_win_pred_func(winX, winY, 'arima', trail_name+str(i), batch_size, seq_len, pred_len, train_ratio)
        p, m = sliding_pred_arima(winY, m)
        pred[i,:] = p

    show_performance(trail_name, pred, real, vis)

    return


def pred_win_hybrid_arima(restr, hi_pred, lo_pred, vis):
    trail_name = "hb_win_%s_%s_%s" %(restr, hi_pred, lo_pred)
    print(colored(trail_name, 'blue'))

    val_num = 10
    win_len = 800
    n_integr = 3

    # slide the window, predict at each step, validate with the last (100) steps
    real = np.zeros((val_num, pred_len))
    pred = np.zeros((val_num, pred_len))
    
    models = [None]*n_integr
 
    for i in tqdm(range(val_num)):
        t = len(dataY) - win_len - pred_len - val_num + i
        real[i,:] = dataY[t+win_len : t+win_len+pred_len, :].reshape(-1)
        winY = dataY[t:t+win_len, :] # (win_len, 1)
        winX = dataX[t:t+win_len, :] # (win_len, n_comp)

        reconstr = series_restr_func(winY, decomp_method=restr, n_integr=n_integr)
        for j in range(n_integr):
            subY = reconstr[j,:].reshape(-1,1)
            _, not_stationary = pm.arima.ADFTest().should_diff(subY)

            # if stationary, goes to high-freq forecast
            if ~not_stationary:
                print('subseq'+str(i)+': high-freq forecast with '+hi_pred+'...')
                p, models[j] = sliding_pred_arima(subY, models[j])
    
            # if non-stationary, goes to low-freq forecast
            else:
                print('subseq'+str(i)+': low-freq forecast with '+lo_pred+'...')
                p = sliding_pred_nn(winX, winY, lo_pred, trail_name+str(i))
                
            pred[i,:] += p.reshape(-1)

    show_performance(trail_name, pred, real, vis)

    return



if __name__ == '__main__':

    # dataX, dataY = loaddata_eu()

    dataX, dataY = loaddata_chn('Guangzhou')

    # pred_nowin_single('tcn', vis=True)
    # pred_nowin_single('gru', vis=True)
    # pred_nowin_single('lstm', vis=True)
    # pred_nowin_single('mlp', vis=True)

    pred_nowin_single('arima', dataX, dataY, vis=True)
    # pred_nowin_single('svr', vis=True)

    # pred_nowin_hybrid('ssa', 'arima', 'arima', vis=True)
    # pred_nowin_hybrid('ssa', 'arima', 'svr', vis=True)
    # pred_nowin_hybrid('ssa', 'arima', 'tcn', vis=True)

    # pred_win_single_arima('arima', True)
    # pred_win_hybrid_arima('ssa', 'arima', 'tcn', True)

    
    