'''
Fit carbon price with SARIMAX model
'''

import itertools
import pickle
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa import stattools # adfuller: stationarity test, q_stat: white noise test
from tqdm import tqdm

from utils import *


def ADF_ACF_PACF(series):
    """ test whether a series is stationary
        print adf test results, plot acf and pacf
        Args:
            series (pd.Series or np.array): the series to analyze
        Returns:
            None
    """
    from statsmodels.tsa.stattools import adfuller # Augmented Dickey-Fuller test
    from statsmodels.graphics.tsaplots import plot_acf # auto correlation function
    from statsmodels.graphics.tsaplots import plot_pacf # partial auto correlation function
    
    # Augmented Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(series, autolag='AIC')  #autolag : {‘AIC’, ‘BIC’, ‘t-stat’, None}
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)
    
    # plot ACF and PACF
    fig = plt.figure(figsize=(15,6))
    ax1 = fig.add_subplot(211)
    fig = sm.graphics.tsa.plot_acf(series, ax=ax1)
    ax2 = fig.add_subplot(212)
    fig = sm.graphics.tsa.plot_pacf(series, ax=ax2)
    plt.show()
    
    return



def sweepSARIMAX(endog, order, seasonal_order=None, exog=None):
    """ determine the best params for SARIMAX model by grid search, calculate AIC and BIC for each param combination
        Args:
            endog (pd.Series or np.array): the series of Y.
            order (tuple: (d,ps,qs)): d (int) is the diff factor; ps (list) is a list of possible p (AR); qs (list) is a list of possible q (MA).
            seasonal_order (tuple: (S,D,Ps,Qs), optional): S (int) is the seasonal period; D (int) is the seasonal diff factor; 
                    Ps (list) is a list of possible P (SAR); Qs (list) is a list of possible Q (SMA). Defaults to None when there is no seasonal effect in the series.
            exog (pd.Series or np.array, optional): the series of X. Defaults to None.
        Notes:
            the list should be consecutive
        Returns:
            aic_mat (np.array): aic_mat[p,q] = aic of (p,d,q), if seasonal_order is None; aic_mat[P,Q,p,q] = aic of (S,P,D,Q,p,d,q), if seasonal_order is not None
            bic_mat (np.array): bic_mat[p,q] = bic of (p,d,q), if seasonal_order is None; bic_mat[P,Q,p,q] = bic of (S,P,D,Q,p,d,q), if seasonal_order is not None
    """
    warnings.simplefilter('ignore')

    # if there is no seasonal effect, i.e., seasonal_order = None
    if not seasonal_order:
        d,ps,qs = order
        bic_mat = np.full((len(ps), len(qs)), np.inf)
        aic_mat = np.full((len(ps), len(qs)), np.inf)
        for p in ps:
            print('\np = %d,q = ' %p)
            # for q in tqdm(qs):
            for q in qs:
                print('%d ' %q, end='')
                if p==0 and q==0:
                    continue
                try:
                    res = sm.tsa.statespace.SARIMAX(endog, exog, order=(p,d,q), seasonal_order=None).fit(disp=False)
                    bic_mat[p,q], aic_mat[p,q] = res.bic, res.aic
                except: # inf
                    continue
            # so far results
            p1,q1 = np.where(aic_mat==np.min(aic_mat))
            p1,q1 = p1[0],q1[0]
            p2,q2 = np.where(bic_mat==np.min(bic_mat))
            p2,q2 = p2[0],q2[0]
            print('\nmin aic for now: %f, corresponding (p,q): (%d,%d)' %(aic_mat[p1,q1], p1,q1))
            print('min bic for now: %f, corresponding (p,q): (%d,%d)' %(bic_mat[p2,q2], p2,q2))
        return aic_mat, bic_mat
    
    # if there is seasonal effect, i.e., seasonal_order is not None TODO not tested
    else:
        d,ps,qs = order
        S,D,Ps,Qs = seasonal_order
        bic_mat = np.full((len(Ps), len(Qs), len(ps), len(qs)), np.inf)
        aic_mat = np.full((len(Ps), len(Qs), len(ps), len(qs)), np.inf)
        for P,Q in itertools.product(Ps, Qs):
            print('\nP,Q = %d,%d, p,q = ' %(P,Q))
            for p,q in itertools.product(ps, qs):
                print('(%d,%d) ' %(p,q), end='')
                try:
                    res = sm.tsa.statespace.SARIMAX(endog, exog, order=(p,d,q), seasonal_order=(P,D,Q,S)).fit(disp=False)
                    bic_mat[P,Q,p,q], aic_mat[P,Q,p,q] = res.bic, res.aic
                except: # inf
                    continue
        return aic_mat, bic_mat



def bestSARIMAX(aic_mat, bic_mat, cri, th=0.2):
    """ determine the best param for SARIMAX model
        Args:
            aic_mat (np.array): the matrix that stores AIC.
            bic_mat (np.array): the matrix that stores BIC.
            cri (str): criterion, 'aic' or 'bic'
            th (float, optional): threshold to filter out the singular cases. Defaults to 0.5.
        Returns:
            order (tuple): (p,q) if dim(mat)==2, (P,Q,p,q) if dim(mat)==4
    """
    mat = np.round(aic_mat) if cri=='aic' else np.round(bic_mat)

    # Sometimes covariance matrix is singular or near-singular, with condition number inf. 
    # This will give a very small bic (~400, while the normal bic is ~4000), and should not be adopted.
    # Threshold should be determined case by case
    mat[np.where( mat<np.median(mat)*th )]=np.inf

    if len(mat.shape) == 2:
        p,q = np.where(mat==np.min(mat))
        p,q = p[0],q[0]
        print('min %s: %f, corresponding (p,q): (%d,%d)' %(cri, mat[p,q], p,q))
        return (p,q)
    else:
        P,Q,p,q = np.where(mat==np.min(mat))
        P,Q,p,q = P[0],Q[0],p[0],q[0]
        print('min %s: %f, corresponding P,Q,p,q: (%d,%d,%d,%d)' %(cri, mat[P,Q,p,q], P,Q,p,q))
        return (P,Q,p,q)



def plot_prediction(model, series):
    """ visualize the results of SARIMAX model
        print model summary, plot results of one-step forecast and dynamic forecast
        Args:
            model (statsmodels.tsa.statespace.sarimax.SARIMAXResultsWrapper): the fitted model
            series (np.array): the observed time series
        Returns:
            None
    """
    # FIXME: ??? FUTURE: if d in the arima model is not 0, pred and real dont have the same length!
    
    print(model.summary())
    
    dy_start = int(len(series)*0.9)
    plot_start = dy_start-40
    plot_end = dy_start+20

    real = series[:]
    # In-sample one-step-ahead predictions
    predict = model.get_prediction()
    # Dynamic predictions
    predict_dy = model.get_prediction(dynamic = dy_start)

    # Graph1
    fig, ax = plt.subplots(figsize=(12,6))
    ax.set(title='SARIMAX', xlabel='Time', ylabel='Price')
    ax.plot(real[plot_start:plot_end], 'o', label='Observed')
    ax.plot(predict.predicted_mean[plot_start:plot_end], 'r--', label='One-step-ahead forecast')
    ci = predict.conf_int()[plot_start:plot_end]
    ax.fill_between(range(plot_end-plot_start), ci[:,0], ci[:,1], color='r', alpha=0.1)
    ax.plot(predict_dy.predicted_mean[plot_start:plot_end], 'g', label='Dynamic forecast')
    ci = predict_dy.conf_int()[plot_start:plot_end]
    ax.fill_between(range(plot_end-plot_start), ci[:,0], ci[:,1], color='g', alpha=0.1)
    legend = ax.legend(loc='best')
    plt.show()
    
    # Graph2
    # FIXME hardcode: the first few samples are not modeled and has a huge error, so we ignore it
    trim = 5
    real = real[trim:]
    pred_in_sample = predict.predicted_mean[trim:]
    pred_out_sample = predict_dy.predicted_mean[trim:]
    rmse_in_sample = cal_rmse(real[:dy_start-trim], pred_in_sample[:dy_start-trim])
    rmse_out_sample = cal_rmse(real[dy_start-trim:], pred_out_sample[dy_start-trim:])
    
    fig, ax = plt.subplots(figsize=(12,6))
    ax.set(title='SARIMAX, in-sample rmse=%.3f, out-sample rmse=%.3f' % (rmse_in_sample, rmse_out_sample), xlabel='Time', ylabel='Price')
    ax.plot(real, label='Observed')
    ax.plot(pred_out_sample, 'green', label='Dynamic forecast', alpha=0.7)
    ax.plot(pred_in_sample, 'orange', label='One-step-ahead forecast', alpha=0.7)
    legend = ax.legend(loc='best')
    plt.show()

    return



def forecast_arima(series, pred_len=1):
    """ fit a series with ARIMA model, differentiate until stationary to determine d, sweeping for the best p,q
        Args:
            series (np.array): the time sequence to be forecasted.
            pred_len (int, optional): the length of prediction. Defaults to 1.
        Returns:
            pred (np.array): the predicted values, length of pred_len
            model (statsmodels.tsa.statespace.sarimax.SARIMAXResultsWrapper): the fitted model 
    """
    warnings.simplefilter('ignore')
    series = series.reshape(-1)
    series_copy = series.copy()

    # # test if the series is a white noise, if yes, return the mean instead of modeling it
    # # 白噪声检验：检验前10个白噪声检验的p值（第二个返回值），所有p值小于0.05，则拒绝原假设，原序列不是白噪声
    # LjungBox = stattools.q_stat(stattools.acf(series)[1:11],len(series))
    # print(LjungBox[1][0])
    # if LjungBox[1][0] > 0.01: # ? any or all? [0] or [-1]?
    #     print('the series is a white noise')
    #     pred = np.mean(series)
    #     return pred, None
    
    # diff until stationary to determine d
    d = 0
    heads = []
    # 平稳性检验：第二个返回值p值，若小于0.05，则拒绝原假设，原序列不存在单位根，即原序列平稳
    while stattools.adfuller(series_copy)[1] > 0.01: # more rigorous
        d += 1
        heads.append(series_copy[0])
        series_copy = np.diff(series_copy)
    
    # sweep for the best params
    ps = range(5)
    qs = range(5)
    aic_mat,bic_mat = sweepSARIMAX(endog=series, order=(d,ps,qs))
    p,q = bestSARIMAX(aic_mat, bic_mat, 'aic')
    
    # fit model
    model = sm.tsa.statespace.SARIMAX(endog=series, order=(p,d,q)).fit(disp=False)
    predict = model.get_prediction(end=model.nobs+pred_len)
    pred = predict.predicted_mean[-pred_len:]

    # # # cumulatively summing back - NOTE: no need to cumsum, since d is already taken in the model
    # # series = np.concatenate((series, [pred]))
    # # while d > 0:
    # #     d -= 1
    # #     series = np.concatenate(([heads[d]], series)).cumsum()
    
    # # sanity check: if the predicted results is unbounded, return the mean
    # pred = series[-1]
    # if np.abs( pred - np.mean(orig) ) > 3 * np.std(orig):
    #     pred = np.mean(orig)

    return pred, model



# NOT recommended, optuna will repeatedly try the same params
# It is better to manually sweep params.
def objective(trial):
    """ objective function in optuna
        Args:
            trail: optuna trail
        Returns:
            bic (float)
    """
    import optuna
    from optuna.trial import TrialState

    warnings.simplefilter('ignore')
    p = trial.suggest_int('p', 1,16)
    q = trial.suggest_int('q', 1,16)
    d = 1
    endog = df.loc[:, 'C_Price']
    exog =  df.loc[:, 'E_Price']

    # Check duplication and skip if it's detected.
    for t in trial.study.get_trials(deepcopy=False, states=[TrialState.COMPLETE]):
        if t.params == trial.params:
            # return t.value  # Return the previous value without re-evaluating it.
            
            # Note that if duplicate parameter sets are suggested too frequently,
            # you can use the pruning mechanism of Optuna to mitigate the problem.
            # By raising `TrialPruned` instead of just returning the previous value,
            # the sampler is more likely to avoid sampling the parameters in the succeeding trials.
            raise optuna.exceptions.TrialPruned()
            
    try:
        res = sm.tsa.statespace.SARIMAX(endog, exog, order=(p,d,q), seasonal_order=None).fit(disp=False)
        # Sometimes covariance matrix is singular or near-singular, with condition number inf. 
        # This will give a very small bic (~400, while the normal bic is ~4000), and should not be adopted.
        # Threshold should be determined case by case
        if res.bic > 3000:  return res.bic
        else:               return np.inf
    except:
        return np.inf


# NOT recommended, optuna will repeatedly try the same params
# It is better to manually sweep params.
def tune_sarimax():
    """ find best params of sarimax with optuna
        Args: None
        Returns: None
    """
    import optuna
    from optuna.trial import TrialState
    
    search_space = {
        'p': range(1,16),
        'q': range(1,16)
    }

    study = optuna.create_study(study_name="sarimax-05251056", direction="minimize", sampler=optuna.samplers.GridSampler(search_space), storage="sqlite:///db.sqlite3", load_if_exists=True)
    # $ optuna-dashboard sqlite:///db.sqlite3
    study.optimize(objective, n_trials=1000, timeout=6000)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    return



if __name__ == "__main__":
    
    df = pd.read_excel('data/df_eu.xlsx', sheet_name='Sheet1')
    dataY = np.array(df['Cprice']).reshape(-1,1)
    dataX = np.array(df.iloc[:,1:]) # carbon price at the previous time step is also used as feature

    # rolling window
    i = 80
    win_len = 500
    val_num = 100
    pred_len = 10
    t = len(dataY) - win_len - pred_len - val_num + i
    win_real = dataY[t+win_len : t+win_len+pred_len] # (pred_len, 1)
    winY = dataY[t:t+win_len, :] 
    winX = dataX[t:t+win_len, :] # (win_len, n_comp)
    
    # NOTE: SARIMAX contains AR terms, so the carbon price at the previous time step is not used as a feature
    # exog = None 
    exog = winX[:,1:]
    endog = winY
    

    # ### Test for stationary
    # print('>>> orig series <<<')
    # ADF_ACF_PACF(dataY)
    # print('>>> diff(1) <<<')
    # ADF_ACF_PACF(np.diff(dataY.reshape(-1)))
    
    
    # ### manually fit the model
    # # sweep for the best params
    # d = 1
    # # grid search
    # p = range(5)
    # q = range(5)
    # orders = (d,p,q)
    # aic_mat, bic_mat = sweepSARIMAX(endog, orders, exog=exog)
    # f = open("sarimax_mat.pkl", "wb")
    # pickle.dump((aic_mat, bic_mat), f)
    # f.close()
    # # f = open("sarimax_mat.pkl", "rb")
    # # aic_mat, bic_mat = pickle.load(f)
    # # f.close()
    # p,q = bestSARIMAX(aic_mat, bic_mat, cri='aic')
    # # visualize
    # model = sm.tsa.statespace.SARIMAX(endog, exog, order=(p,d,q)).fit(disp=False)
    # print(model.summary())
    # plot_prediction(model, endog)
    
    pred, model = forecast_arima(endog,pred_len)
    print(pred)
    print(win_real)
    # forecast_sarimax(endog, exog)


    ### optune for the best params - NOT recommended
    # tune_sarimax()
    