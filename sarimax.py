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
            aic_mat (tuple): training set   (feature as np.array, label as np.array)
            bic_mat (tuple): validation set (feature as np.array, label as np.array)
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
            print('min aic for now: %f, corresponding (p,q): (%d,%d)' %(aic_mat[p1,q1], p1,q1))
            print('min bic for now: %f, corresponding (p,q): (%d,%d)' %(bic_mat[p2,q2], p2,q2))
        return aic_mat, bic_mat
    
    # if there is seasonal effect, i.e., seasonal_order is not None
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



def bestSARIMAX(aic_mat, bic_mat, cri):
    """ determine the best param for SARIMAX model
        Args:
            aic_mat (np.array): the matrix that stores AIC.
            bic_mat (np.array): the matrix that stores BIC.
            cri (str): criterion, 'aic' or 'bic'
        Returns:
            order (tuple): (p,q) if dim(mat)==2, (P,Q,p,q) if dim(mat)==4
    """
    mat = np.round(aic_mat) if cri=='aic' else np.round(bic_mat)

    # Sometimes covariance matrix is singular or near-singular, with condition number inf. 
    # This will give a very small bic (~400, while the normal bic is ~4000), and should not be adopted.
    # Threshold should be determined case by case
    # mat[np.where( mat<np.median(mat)*0.7 )]=np.inf

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



def forecast_arima(series):
    """ fit a series with ARIMA model, differentiate until stationary to determine d, sweeping for the best p,q
        Args:
            series (np.array): the time sequence to be forecasted.
        Returns:
            series[-1] (float): the next time step
            model (statsmodels.tsa.statespace.sarimax.SARIMAXResultsWrapper): the fitted model 
    """
    orig = series.copy()

    # test if the series is a white noise, if yes, return a random number instead of modeling it
    # 白噪声检验：检验前10个白噪声检验的p值（第二个返回值），所有p值小于0.05，则拒绝原假设，原序列不是白噪声
    LjungBox = stattools.q_stat(stattools.acf(series)[1:11],len(series))
    print(LjungBox[1][0])
    if LjungBox[1][0] > 0.01: # ? any or all? [0] or [-1]?
        print('the series is a white noise')
        pred = np.mean(series) + np.std(series) * np.random.rand(1)
        return pred, None
    
    # diff until stationary to determine d
    d = 0
    heads = []
    # 平稳性检验：第二个返回值p值，若小于0.05，则拒绝原假设，原序列不存在单位根，即原序列平稳
    while stattools.adfuller(series)[1] > 0.01: # more rigorous
        d += 1
        heads.append(series[0])
        series = np.diff(series)
    
    # sweep for the best params
    ps = range(5)
    qs = range(5)
    aic_mat,bic_mat = sweepSARIMAX(endog=series, order=(d,ps,qs))
    p,q = bestSARIMAX(aic_mat, bic_mat, 'bic')
    warnings.simplefilter('ignore')
    model = sm.tsa.statespace.SARIMAX(endog=series, order=(p,d,q)).fit(disp=False)
    pred = model.forecast()[0]

    # cumulatively summing back
    series = np.concatenate((series, [pred]))
    while d > 0:
        d -= 1
        series = np.concatenate(([heads[d]], series)).cumsum()
    
    # sanity check
    pred = series[-1]
    if np.abs( pred - np.mean(orig) ) > 5 * np.std(orig):
        pred = 0

    print(series.shape)

    return pred, model



def plot_prediction(model, series):
    """ visualize the results of SARIMAX model
        print model summary, plot results of one-step forecast and dynamic forecast
        Args:
            model (statsmodels.tsa.statespace.sarimax.SARIMAXResultsWrapper): the fitted model
            series (np.array): the observed time series
        Returns:
            None
    """
    # FIXME: if d in the arima model is not 0, pred and real dont have the same length!
    print(model.summary())

    # In-sample one-step-ahead predictions
    predict = model.get_prediction()
    # Dynamic predictions
    predict_dy = model.get_prediction(dynamic = int(len(series)*0.8))

    real = series.copy()
    pred = predict.predicted_mean
    print(real.shape)
    print(pred.shape)
    rmse = np.sqrt(np.mean( np.square(real-pred) ))

    # Graph1
    fig, ax = plt.subplots(figsize=(12,6))
    ax.set(title='High-freq forecast with ARIMA', xlabel='Time', ylabel='Price')
    plot_start = int(len(series)*0.7)
    plot_end = len(series)-1
    ax.plot(real[plot_start:plot_end], 'o', label='Observed')
    ax.plot(predict.predicted_mean[plot_start:plot_end], 'r--', label='One-step-ahead forecast')
    ci = predict.conf_int()[plot_start:plot_end]
    ax.fill_between(range(plot_end-plot_start), ci[:,0], ci[:,1], color='r', alpha=0.1)
    ax.plot(predict_dy.predicted_mean[plot_start:plot_end], 'g', label='Dynamic forecast')
    ci = predict_dy.conf_int()[plot_start:plot_end]
    ax.fill_between(range(plot_end-plot_start), ci[:,0], ci[:,1], color='g', alpha=0.1)
    legend = ax.legend(loc='best')
    
    # Graph2
    fig, ax = plt.subplots(figsize=(12,6))
    ax.set(title='High-freq component, RMSE=%.3f'%rmse, xlabel='Time', ylabel='Price')
    ax.plot(predict_dy.predicted_mean, 'g', label='Dynamic forecast')
    ax.plot(real, label='Observed')
    ax.plot(pred, label='One-step-ahead forecast')
    legend = ax.legend(loc='best')
    fig.show()

    return



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
    
    df = pd.read_excel('./testdata/CCprice.xlsx', sheet_name='Sheet1')
    df = df.set_index('Date')
    df.index = pd.DatetimeIndex(df.index)
    df.index.freq = df.index.inferred_freq
    date_col = df.index

    # # overview of data
    # print(df.info())
    # fig = plt.figure()
    # ax1 = fig.add_subplot(111)
    # ax1.plot(date_col, df['C_Price'], 'r', label='C_price')
    # ax1.legend(loc=2)
    # ax1.set_ylabel('C_Price')
    # ax2 = ax1.twinx() # this is the important function
    # ax2.plot(date_col, df['E_Price'], 'g', label='E_price')
    # ax2.legend(loc=1)
    # ax2.set_ylabel('E_price')
    # plt.show()

    # # Test for stationary
    # print('>>> orig series <<<')
    # ADF_ACF_PACF(df['C_Price'])
    # print('>>> diff(1) <<<')
    # ADF_ACF_PACF(df['C_Price'].diff(1).dropna())

    # sweep for the best params
    endog = df.loc[:, 'C_Price']
    exog =  df.loc[:, 'E_Price']
    d = 1
    # # grid search
    # p = range(21)
    # q = range(21)
    # order = (d,p,q)
    # aic_mat,bic_mat = sweepSARIMAX(endog, order)
    # f = open("mat_sarimax2.pkl", "wb")
    # pickle.dump((aic_mat, bic_mat), f)
    # f.close()
    # or read from file
    f = open("mat_sarimax2.pkl", "rb")
    aic_mat, bic_mat = pickle.load(f)
    f.close()
    p,q = bestSARIMAX(aic_mat, bic_mat, 'aic')
    model = sm.tsa.statespace.SARIMAX(endog, order=(p,d,q)).fit(disp=False)
    print(model.summary())

    # plotPrediction(model)

    # optune for the best params
    # tune_sarimax()