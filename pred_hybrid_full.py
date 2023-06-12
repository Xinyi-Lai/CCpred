# TODO:!!!!!!

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import HiddenPrints


# step1: decompose into subsequences
from series_restr import decomp_ceemdan, decomp_eemd, decomp_emd, decomp_ewt

# step2: integrate subsequences into hi-freq and lo-freq
from series_restr import integr_fuzzen_threshold, integr_fine_to_coarse

# step3: high-freq forecast with ARIMA, in sarimax.py
from sarimax import forecast_arima
  
# step4: low-freq forecast, in hybrid_model.py
from nn_models import forecast_BPNN, forecast_LSTM, forecast_GRU, forecast_TCN


if __name__ == '__main__':

    params = {
        'win_len': 200,
        'win_step': 1,
        'decompose_method': 'ceemdan', # {'ceemdan', 'eemd', 'emd', 'ewt', 'ssa'}
        'integrate_method': 'fuzzen',
        'hi_pred_method': 'arima',
        'lo_pred_method': 'tcn', # {'bpnn', 'lstm', 'gru', 'tcn'}
        'error_correction': False
    }
    trail_name = "hb_win%d_sam%d_%s_%s_%s_%s_%s" %(
        params['win_len'], params['win_step'], params['decompose_method'], params['integrate_method'], 
        params['hi_pred_method'], params['lo_pred_method'], 'ec' if params['error_correction'] else '')
    print(trail_name)

    # load data
    df = pd.read_excel('./testdata/CCprice.xlsx', sheet_name='Sheet1')
    Cprice = np.array(df['C_Price'])[-1500:]

    # slide the window, and predict at each step
    pred = []
    real = []
    errors = []
    win_len = params['win_len']
    win_step = params['win_step']
    for t in tqdm(range(0, len(Cprice)-win_len-1, win_step)): # sample!!!
        print('\n step: %i' %t, end='')
        win_x = Cprice[t:t+win_len]
        win_y = Cprice[t+win_len]

        print(' -> sequences restructuring', end='')
        if params['decompose_method'] == 'ceemdan':
            imfs = decomp_ceemdan(win_x)
        elif params['decompose_method'] == 'eemd':
            imfs = decomp_eemd(win_x)
        elif params['decompose_method'] == 'emd':
            imfs = decomp_emd(win_x)
        elif params['decompose_method'] == 'ewt':
            imfs = decomp_ewt(win_x, num_comp=10)
        elif params['decompose_method'] == 'ssa':
            imfs = decomp_ssa(win_x, num_comp=10)
        else:
            print('unrecognized decompose_method: '+params['decompose_method'])
            break

        if params['integrate_method'] == 'fuzzen':
            hi_freq, lo_freq, _ = integr_fuzzen(imfs)
        else:
            print('unrecognized integrate_method: '+params['integrate_method'])
            break

        print(' -> high-freq forecasting', end='')
        if params['hi_pred_method'] == 'arima':
            try:
                with HiddenPrints():
                    hi_freq_pred, _ = forecast_arima(hi_freq)
            except Exception as e:
                hi_freq_pred = np.mean(hi_freq)
                print('error in hi_pred: ',e.__class__.__name__,e)
        else:
            print('unrecognized hi_pred_method: '+params['hi_pred_method'])
            break
        
        print(' -> low-freq forecasting', end='')
        with HiddenPrints():
            if params['lo_pred_method'] == 'bpnn':
                lo_freq_pred, _ = forecast_BPNN(lo_freq, trail_name)
            elif params['lo_pred_method'] == 'lstm':
                lo_freq_pred, _ = forecast_LSTM(lo_freq, trail_name)
            elif params['lo_pred_method'] == 'gru':
                lo_freq_pred, _ = forecast_GRU(lo_freq, trail_name)
            elif params['lo_pred_method'] == 'tcn':
                lo_freq_pred, _ = forecast_TCN(lo_freq, trail_name)
            else:
                print('unrecognized lo_pred_method: '+params['lo_pred_method'])
                break

        error_pred = 0
        if params['error_correction']:
            print(' -> error correction ', end='')
            if len(errors) > 50:
                try:
                    with HiddenPrints():
                        error_pred, _ = forecast_arima(np.array(errors))
                        # error_pred, _ = forecast_TCN(np.array(errors), 'error_tcn')
                except Exception as e:
                    error_pred = 0
                    print('错误明细是',e.__class__.__name__,e)
            print(error_pred)
        
        step_pred = hi_freq_pred + lo_freq_pred + error_pred
        pred.append(step_pred)
        real.append(win_y)
        errors.append(win_y-step_pred)
        if len(errors) > 100: # throw away errors long time ago
            del errors[0]

        print(' -> predicted %.3f, observed %.3f' %(step_pred, win_y))
    
    pred = np.array(pred)
    real = np.array(real)

    # store
    f = open(trail_name+".pkl", "wb")
    pickle.dump((params, pred, real), f)
    f.close()

    # # load
    # f = open(trail_name+".pkl", "rb")
    # params, pred, real = pickle.load(f)
    # f.close()

    # print(pred)
    # print(real)

    # visualize
    rmse = np.sqrt(np.mean( np.square(real-pred) ))
    mape = np.mean(np.abs(real-pred)/real)*100
    plt.figure()
    plt.title('%s, RMSE=%.2f, MAPE=%.2f%%' %(trail_name,rmse,mape))
    plt.plot(pred, label='pred')
    plt.plot(real, label='real')
    plt.legend()
    plt.show()