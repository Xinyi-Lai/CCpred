'''
For the report, predict with hybrid framework (no windowing, restructure + pred)
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


# load data
df = pd.read_excel('data/df_eu.xlsx', sheet_name='df-eu')
Xcols = ['Eprice', 'BrentOil', 'CrudeOilF', 'TTF-NatGas', 'NatGasF', 'Coal', 'GasolineF', 'DJI', 'S&P500', 'USD-EUR']
dataX = np.array(df[Xcols]) 
dataY = np.array(df['Cprice']).reshape(-1,1)
# print(dataX.shape)
# print(dataY.shape)

reconstr = restr_ssa(dataY.reshape(-1), n_decomp=10, n_integr=5, vis=False)

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


# m = TCN_model('sg-nowin-TCN', batch_size=batch_size)
# m = LSTM_model('sg-nowin-LSTM', batch_size=batch_size)
# m = GRU_model('sg-nowin-GRU', batch_size=batch_size)
# m = BPNN_model('sg-nowin-BPNN', batch_size=batch_size)



