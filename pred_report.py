'''
For the report, no windowing, predict with single and hybrid framework
'''

import pandas as pd
import numpy as np
from torchinfo import summary

from models import *
from series_restr import *
from utils import *


batch_size = 1
pred_len = 10
seq_len = 100

### load data
df = pd.read_excel('data/df_eu.xlsx', sheet_name='df-eu')
Xcols = ['Cprice', 'Eprice', 'BrentOil', 'CrudeOilF', 'TTF-NatGas', 'NatGasF', 'Coal', 'GasolineF', 'DJI', 'S&P500', 'USD-EUR']
# note: Cprice at previous steps are also features for the current step
dataX = np.array(df[Xcols]) 
dataY = np.array(df['Cprice']).reshape(-1,1)
# print(dataX.shape)
# print(dataY.shape)


# no windowing, single framework, predict with single model
def pred_nowin_single(method, vis=True):
    method_dict = { 'tcn': TCN_model, 'gru': GRU_model, 'lstm': LSTM_model, 'bpnn': BPNN_model } # 'seq2seq+': Seq2SeqPlus_model, 'seq2seq': Seq2Seq_model, 
    if method not in method_dict.keys():
        print('unrecognized method: ' + method)
        return
    
    trail_name = "sg_nowin_%s" %(method)
    print(colored(trail_name, 'blue'))

    m = method_dict[method](trail_name, batch_size=batch_size)
    m.prepare_data(dataX, dataY, seq_len=seq_len, pred_len=pred_len)
    m.init_model()
    m.train_model(train_ratio=0.9, val_ratio=0.2) # FIXME train_ratio=0.8
    # m.load_model()
    # m.print_summary()
    pred, real = m.test_model(test_ratio=0.1) # FIXME test_ratio=0.2
    print(pred.shape)
    show_performance(trail_name, pred, real, vis)
    return


def pred_nowin_single_arima():

    trail_name = "sg_nowin_arima"
    print(colored(trail_name, 'blue'))

    test_ratio = 0.1

    split = int(len(self.dataset)*(1-test_ratio))
    Dte = self.dataset[split:]
    Dte = DataLoader(MyDataset(Dte), batch_size=self.batch_size)
    

    return


# no windowing, hybrid framework, restructure + predict + sum
def pred_nowin_hybrid():

    reconstr = restr_ssa(dataY.reshape(-1), n_decomp=10, n_integr=3, vis=False)
    pred = []
    real = []

    for i in range(reconstr.shape[0]):
        print(i)
        subY = reconstr[i,:].reshape(-1,1)
        m = TCN_model('hb-nowin-TCN'+str(i), batch_size)
        m.prepare_data(dataX, subY, seq_len=seq_len, pred_len=pred_len)
        m.init_model()
        m.train_model(train_ratio=1, val_ratio=0.2)
        # m.load_model()
        sub_pred, sub_real = m.test_model(test_ratio=1)
        pred.append(sub_pred)
        real.append(sub_real)

    pred = np.sum(np.array(pred), axis=0)
    real = np.sum(np.array(real), axis=0)
    show_performance('hb-nowin-TCN', pred, real, vis=True)

    return


if __name__ == '__main__':

    pred_nowin_single('tcn', vis=True)

    
    