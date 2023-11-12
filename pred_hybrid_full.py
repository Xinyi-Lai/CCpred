'''
!!! not tested yet !!!
'''

import pickle
import pandas as pd
import numpy as np

from statsmodels.tsa import stattools
from termcolor import colored
from tqdm import tqdm

from utils import *
from nn_models import *
from series_restr import *
from sarimax import forecast_arima

import random
import torch

# TODO Fix the random seed to ensure the reproducibility of the experiment
random_seed = 10
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)


batch_size = 1



### Predict with hybrid framework. full process (window + restructure + pred)
def pred_full(win_len, restr, hi_pred, lo_pred, seq_len=100, pred_len=10, vis=False, val_num=100):

    params = {
        'win_len': win_len, # {800, 500}
        'restr': restr, # {'ssa', 'ssa_ex', 'ceemdan', 'ceemdan_ex', 'eemd', 'emd'}
        'hi_pred': hi_pred, # {'tcn', 'gru', 'lstm', 'bpnn', TODO 'arima'}
        'lo_pred': lo_pred, # {'tcn', 'gru', 'lstm', 'bpnn'}
        'seq_len': seq_len,
        'pred_len': pred_len
    }
    trail_name = 'fu_win%d_%s_%s_%s' %(win_len, restr, hi_pred, lo_pred)
    print(colored(trail_name, 'blue'))


    model_dict = {
        'tcn': TCN_model, 'gru': GRU_model, 'lstm': LSTM_model, 'bpnn': BPNN_model # 'arima': forecast_arima,
    }
    if lo_pred not in model_dict.keys():
        print('unrecognized lo_pred_method: '+lo_pred)
        return
    if hi_pred not in model_dict.keys():
        print('unrecognized hi_pred_method: '+hi_pred)
        return
    
    restr_dict = {
        'ssa': restr_ssa,           'ssa_ex':restr_ssa_ex, 
        'ceemdan': decomp_ceemdan,  'ceemdan_ex': decomp_ceemdan_ex,
        'eemd': decomp_eemd,        'eemd_ex': decomp_eemd_ex,
        'emd': decomp_emd,          'emd_ex': decomp_emd_ex
    }
    if restr not in restr_dict.keys():
        print('unrecognized restr: ' + restr)
        return


    # load data
    df = pd.read_excel('data\df.xlsx', sheet_name='Sheet1')
    dataY = np.array(df['Cprice'])
    dataX = np.array(df.iloc[:,1:]) # carbon price at the previous time step is also used as feature

    # hold results
    real = np.zeros((val_num, pred_len))
    pred = np.zeros((val_num, pred_len))
    # slide the window, decompose and predict at each step, validate for the last (100) steps
    for ii in tqdm(range(val_num)):
        t = len(dataY) - win_len - pred_len - val_num + ii
        win_x = dataX[t:t+win_len, :] # Xvars, (win_len, n_comp)
        series = dataY[t:t+win_len] # Yvar, (win_len,)
        real[ii,:] = dataY[t+win_len : t+win_len+pred_len] # target, (pred_len,)
        
        # decompose
        restructured = restr_dict[restr](series)
        if restr not in ['ssa_ex', 'ssa']:
            restructured = integr_fuzzen_pwlf(restructured, n_integr=3)
        
        # predict
        # for each subsequence
        for i in range(restructured.shape[0]):
            sub_seq = restructured[i].reshape(-1,1)

            # if white noise, return the mean
            if stattools.q_stat(stattools.acf(sub_seq)[1:11],len(sub_seq))[1][0] > 0.01:
                sub_pred = np.mean(sub_seq)

            # if stationary, goes to high-freq forecast
            elif stattools.adfuller(sub_seq)[1] < 0.01:
                m = model_dict[hi_pred](trail_name+str(i), batch_size)
                try:
                    with HiddenPrints():
                        sub_pred = m.predict(win_x, sub_seq, seq_len, pred_len)
                except Exception as e:
                    sub_pred = sub_seq[-1] # take the last step as prediction
                    print('error in hi_pred: ', hi_pred, e.__class__.__name__, e)
                
            # if non-stationary, goes to low-freq forecast
            else:
                m = model_dict[lo_pred](trail_name+str(i), batch_size)
                try:
                    with HiddenPrints():
                        sub_pred = m.predict(win_x, sub_seq, seq_len, pred_len)
                except Exception as e:
                    sub_pred = sub_seq[-1] # take the last step as prediction
                    print('error in lo_pred: ', lo_pred, e.__class__.__name__, e)
                    
            # integrate the subsequence prediction
            pred[ii,:] += sub_pred.reshape(-1)


    # store
    f = open(trail_name+".pkl", "wb")
    pickle.dump((params, pred, real), f)
    f.close()

    # # load
    # f = open("results\\"+trail_name+".pkl", "rb")
    # _, pred, real = pickle.load(f)
    # f.close()

    # performance
    show_performance(trail_name, pred, real, vis)

    return



### Predict with decomposed data (windowing + restructuring + [multi-feature pred]).
def pred_decomp(win_len, win_step, restr, method, vis=False):

    params = {
        'win_len': win_len,
        'win_step': win_step,
        'restr': restr, # {'ceemdan', 'eemd', 'emd', 'ssa'}
        'method': method, # {'bpnn', 'lstm', 'gru', 'tcn'}
    }
    trail_name = 'dc_win%d_sam%d_%s_%s' %(params['win_len'], params['win_step'], params['restr'], params['method'])
    print(colored(trail_name, 'blue'))


    # load windowed and restructured data 
    # load data of win_step=1, params['win_step'] is used to control the simulation progress )
    decomp_datapath = 'data\\restructured\\restr_win%d_sam1_%s.pkl' %(params['win_len'], params['restr'])
    f = open(decomp_datapath, "rb")
    _, win_restrs, win_ys = pickle.load(f)
    f.close()
    
    # only using the last 1000 samples for simulation FIXME
    # win_restrs = win_restrs[-1000:]
    # win_ys = win_ys[-1000:]
    print(win_restrs.shape) # (1000,5,200) (1000,n_comp,win_len)

    # slide the window, and predict at each step
    timesteps = range(0, win_restrs.shape[0], win_step) # sample!
    real = np.zeros(len(timesteps))
    pred = np.zeros(len(timesteps))
    
    for t in tqdm(range(len(timesteps))):
        # print('\n step: %i' %timesteps[t], end='')
        real[t] = win_ys[timesteps[t]]
        restr = win_restrs[timesteps[t]]

        if t == 0:
            # take all subsequences as input features
            with HiddenPrints():
                if params['method'] == 'tcn':
                    pred[t], _ = forecast_TCN_decomp(restr, trail_name)
                elif params['method'] == 'gru':
                    pred[t], _ = forecast_GRU_decomp(restr, trail_name)
                # elif params['method'] == 'bpnn':
                #     sub_pred, _ = forecast_BPNN(sub_seq, trail_name)
                # elif params['method'] == 'lstm':
                #     sub_pred, _ = forecast_LSTM(sub_seq, trail_name)
                else:
                    print('unrecognized method: '+params['method'])
                    return
        else:
            pred[t] = real[t-1]

        # print('step %i -> predicted %.3f, observed %.3f' %(timesteps[t], pred[t], real[t]))
        

    # store
    f = open(trail_name+'.pkl', 'wb')
    pickle.dump((params, pred, real), f)
    f.close()

    # # load
    # f = open(trail_name+'.pkl', 'rb')
    # params, pred, real = pickle.load(f)
    # f.close()

    rmse = cal_rmse(real, pred)
    mape = cal_mape(real, pred)
    print(colored('%s, RMSE=%.2f, MAPE=%.2f%%' %(trail_name, rmse, mape), 'green'))

    # visualize
    if vis:
        plt.figure()
        plt.title('%s, RMSE=%.2f, MAPE=%.2f%%' %(trail_name, rmse, mape))
        plt.plot(pred, label='pred')
        plt.plot(real, label='real')
        plt.legend()
        plt.show()

    return



### Predict error series (windowing + restructuring + pred + [error correction]).
def pred_hybrid_ec():

    # # step1: decompose into subsequences
    # # step2: integrate subsequences into hi-freq and lo-freq
    # from series_restr import *

    # step3: high-freq forecast with ARIMA, in sarimax.py
    # from sarimax import forecast_arima

    # step4: low-freq forecast with neural networks, in nn_models.py
    # from nn_models import forecast_BPNN, forecast_LSTM, forecast_GRU, forecast_TCN

    # step5: error correction, predict error series
    from sarimax import forecast_arima
    from nn_models import forecast_TCN

    params = {
        'win_len': 200,
        'win_step': 1,
        'restr': 'ssa', # {'ceemdan', 'eemd', 'emd', 'ewt', 'ssa'}
        'hi_pred': 'arima', # {''}
        'lo_pred': 'tcn', # {'bpnn', 'lstm', 'gru', 'tcn'}
        'error_correction': True
    }
    trail_name = 'hb_win%d_sam%d_%s_%s_%s_%s' %(
        params['win_len'], params['win_step'], params['restr'], 
        params['hi_pred'], params['lo_pred'], 'ec' if params['error_correction'] else '')
    print(trail_name)


    # load trained data
    pretrained_file = 'hb_win%d_sam%d_%s_%s_%s.pkl' %(params['win_len'], params['win_step'], params['restr'], params['hi_pred'], params['lo_pred'])
    print(pretrained_file)
    if not os.path.exists(pretrained_file):
        print('pretrained data not available')
        return

    f = open(pretrained_file, "rb")
    _, pred, real = pickle.load(f)
    f.close()
    
    # slide the window, load sub_pred from window, perform error correction 
    pred_ec = np.zeros(real.shape)
    errors = np.zeros(100) # store errors for dynamic error correction
    for t in tqdm(range(len(real))):

        step_pred = np.sum(pred[:,t])
        win_y = real[t]

        error_pred = 0
        if t > 100:
            try:
                with HiddenPrints():
                    error_pred, _ = forecast_arima(errors)
                    # error_pred, _ = forecast_TCN(errors, 'error_tcn')
            except Exception as e:
                error_pred = 0
                # print('step %i'%t)
                # print(errors)
                print('error in ec: ',e.__class__.__name__,e)
        step_pred += error_pred

        # step ends
        errors = np.roll(errors,-1) # shift left by one, throw away errors long time ago
        errors[-1] = win_y-step_pred
        pred_ec[t] = step_pred
        # print(' -> predicted %.3f, observed %.3f, ec %.3f' %(step_pred, win_y, error_pred))
        

    # store
    f = open(trail_name+".pkl", "wb")
    pickle.dump((params, pred, real, pred_ec), f)
    f.close()

    # # load
    # f = open(trail_name+".pkl", "rb")
    # params, pred, real, pred_ec = pickle.load(f)
    # f.close()


    # visualize
 
    rmse = np.sqrt(np.mean( np.square(real-pred_ec) ))
    mape = np.mean(np.abs(real-pred_ec)/real)*100
    plt.figure()
    plt.title('%s, RMSE=%.2f, MAPE=%.2f%%' %(trail_name,rmse,mape))
    plt.plot(pred_ec, label='pred')
    plt.plot(real, label='real')
    plt.legend()
    plt.show()

    return



if __name__ == '__main__':

    pred_full(win_len=500, restr='ceemdan', hi_pred='bpnn', lo_pred='tcn')


    # params = {
    #     'win_len': win_len,
    #     'win_step': win_step,
    #     'restr': restr, # {'ssa', 'ssa_ex', 'ceemdan', 'ceemdan_ex', 'eemd', 'emd', }
    #     'method': method, # {'bpnn', 'lstm', 'gru', 'tcn'}
    # }
    # pred_decomp(win_len=200, win_step=1, restr='ssa', method='tcn', vis=True)


    # pred_hybrid_ec()