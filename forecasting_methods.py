''' forecasting methods
'''

import numpy as np
from tqdm import tqdm
from utils import *
import warnings

# for svr
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, cross_val_score

# for arima
import pmdarima as pm
from copy import deepcopy

########## machine learning models ##########


def ml_prepare_data(data, seq_len=100, pred_len=10):
    ''' prepare data for machine learning models
        Args:
            data (np.array), shape (n_samples, 1+n_features): target at the first column, followed by other features 
            seq_len (int): how many previous targets considered as features also (sequence length in RNN). Defaults to 100.
            pred_len (int): num of steps to predict. Defaults to 10.
        Returns:
            ml_x (np.array), shape (n_samples-seq_len-pred_len+1, seq_len+n_features): features
            ml_y (np.array), shape (n_samples-seq_len-pred_len+1, pred_len): targets
    '''
    n_samples = data.shape[0]
    ml_x, ml_y = [], []
    for i in range(n_samples-seq_len-pred_len+1):
        xi = np.concatenate((data[i:i+seq_len-1, 0].reshape(-1), data[i+seq_len-1, :].reshape(-1) ) ) # seq_len-1 previous targets + current features
        yi = data[i+seq_len : i+seq_len+pred_len, 0].reshape(-1) # the first col is target
        ml_x.append(xi)
        ml_y.append(yi)
    ml_x, ml_y = np.array(ml_x), np.array(ml_y)
    # print(ml_x.shape, ml_y.shape)
    return ml_x, ml_y


### SVR (support vector regression)
def pred_svr(ml_x, ml_y, train_ratio=0.9, auto=False):
    ''' predict with SVR (Support Vector Regression)
        Args:
            ml_x (np.array), shape (n_samples, n_features): features
            ml_y (np.array), shape (n_samples, pred_len): targets
            train_ratio (float): ratio of training data. Defaults to 0.9.
            auto (bool): whether to auto-tune parameters. Defaults to False.
        Returns:
            real (np.array), shape (n_samples, pred_len): real targets
            pred (np.array), shape (n_samples, pred_len): predicted targets
            # score (np.array), shape (pred_len,): scores for each step (1/rmse)
    '''

    split = int(len(ml_y)*train_ratio)
    test_size = len(ml_y) - split # some rounding issues
    real = np.zeros((test_size, ml_y.shape[1]))
    pred = np.zeros((test_size, ml_y.shape[1]))
    score = np.zeros(ml_y.shape[1])

    # predict each step (each y in ml_y)
    for i in tqdm(range(ml_y.shape[1])):
        x, y = ml_x, ml_y[:,i]
        scalarX, scalarY = StandardScaler(), StandardScaler()
        x = scalarX.fit_transform(x)
        y = scalarY.fit_transform(y.reshape(-1,1)).reshape(-1)
        train_x, test_x = x[:split], x[split:]
        train_y, test_y = y[:split], y[split:]

        clf = SVR( kernel='linear', C=0.1 ) # baseline
        
        if auto: # auto-tune parameters, not always better
            baseline_score = np.mean(cross_val_score(clf, train_x, train_y, cv=5, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1))
            param_grid = {
                'kernel': ['rbf', 'linear', 'poly', 'sigmoid'], 
                'C': [pow(10,i) for i in range(-2,3)],
                'gamma': ['scale','auto'] + [pow(10,i) for i in range(-5,-1)], 
            }
            random_search = RandomizedSearchCV(SVR(), param_grid, cv=5, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)
            random_search.fit(train_x, train_y)
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

        clf.fit(train_x, train_y)

        # calculate score on recent steps in training set
        train_p = clf.predict(train_x)
        train_p = scalarY.inverse_transform(train_p.reshape(-1,1)).reshape(-1)
        train_y = scalarY.inverse_transform(train_y.reshape(-1,1)).reshape(-1)
        score[i] = 1 / np.sqrt(cal_rmse(train_y, train_p))

        # make predictions on the test set
        test_p = clf.predict(test_x)
        pred[:,i] = scalarY.inverse_transform(test_p.reshape(-1,1)).reshape(-1)
        real[:,i] = scalarY.inverse_transform(test_y.reshape(-1,1)).reshape(-1)

    return pred, real#, score




########## time series analysis models ##########


def pred_arima(data, seq_len=100, pred_len=10, train_ratio=0.9, auto=True, order=(5,3,5)):
    ''' predict with ARIMA (AutoRegressive Integrated Moving Average)
        Args:
            data (np.array), shape (n_samples, 1+n_features): target at the first column, followed by other features 
            seq_len (int): sequence length in RNN, for alignment. Defaults to 100.
            pred_len (int): num of steps to predict. Defaults to 10.
            train_ratio (float): ratio of training data. Defaults to 0.9.
            auto (bool): whether to auto-tune parameters. Defaults to False.
        Returns:
            real (np.array), shape (n_samples, pred_len): real targets
            pred (np.array), shape (n_samples, pred_len): predicted targets
            # score (np.array), shape (pred_len,): scores fitted sequence (1/rmse)
    '''
    warnings.simplefilter('ignore') # ignore warnings
    dataY = data[:,0]
    split = int( (len(dataY)-seq_len-pred_len+1) * train_ratio )  + seq_len # the first seq is in training set
    trainY, testY = dataY[:split], dataY[split:]

    # fit model with training set

    if auto: # auto-tune parameters, better    
        model = pm.auto_arima(  
            trainY, start_p=1, start_q=1,
            max_p=10, max_q=10,         # maximum p and q
            information_criterion='aic',# 'aic', 'aicc', 'bic', 'hqic', 'oob'
            d=None,                     # d=None let model determine 'd' # FIXME
            test='adf',                 # use adftest to find optimal 'd', or 'kpss', 'pp'
            m=1,                        # frequency of series, m=1 means non-seasonal
            seasonal=False,             # no Seasonality
            stepwise=True               # The stepwise algorithm can be significantly faster than a non-stepwise selection (i.e., essentially a grid search) and is less likely to over-fit the model. 
        ) 
    else: # manually set parameters, baseline
        model = pm.arima.ARIMA(order=order) # baseline
        model.fit(trainY)

    # print(model.summary())

    # calculate the fitting score on the training set
    fitted = model.fittedvalues()
    score = 1 / np.sqrt(cal_rmse(trainY, fitted))
    score = np.array([score] * pred_len) # FIXME

    # make predictions on the test set
    pred = np.zeros((len(testY)-pred_len+1, pred_len))
    real = np.zeros((len(testY)-pred_len+1, pred_len))
    for i in tqdm(range(len(testY)-pred_len+1)):
        model_copy = deepcopy(model)
        if i != 0: model_copy.update(testY[:i], maxiter=0) # NOTE: actually still windowing effect, parameters are updated with new observations
        new_pred = model_copy.predict(n_periods=pred_len, return_conf_int=False)
        pred[i,:] = new_pred
        real[i,:] = dataY[split+i: split+i+pred_len].reshape(-1)

    return pred, real#, score