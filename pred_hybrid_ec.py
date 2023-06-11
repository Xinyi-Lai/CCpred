'''
Predict error series (windowing + restructuring + pred + [error correction]).
'''
# TODO:!!!!!!
import matplotlib.pyplot as plt
import numpy as np
import pickle
from tqdm import tqdm

from method import HiddenPrints


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


def main():

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

    main()