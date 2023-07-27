'''
Predict with single framework (window + pred).
FUTURE: support arima??
'''

import numpy as np
import pandas as pd
import pickle
import random
from termcolor import colored
import torch
from tqdm import tqdm

from utils import *
from models import *


# TODO Fix the random seed to ensure the reproducibility of the experiment
random_seed = 10
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)


batch_size = 8


# test nn models
def pred_step(win_len, t):
    df = pd.read_excel('data\df.xlsx', sheet_name='Sheet1')
    dataY = np.array(df['Cprice'])
    dataX = np.array(df.iloc[:,1:]) # carbon price at the previous time step is also used as feature

    dataY = dataY[t:t+win_len, np.newaxis] # (win_len, 1), instead of (win_len,)
    dataX = dataX[t:t+win_len, :] # (win_len, n_comp)

    # m1 = TCN_model('step-TCN', batch_size)
    # m2 = GRU_model('step-GRU', batch_size)
    # m3 = LSTM_model('step-LSTM', batch_size)
    # m4 = BPNN_model('step-BPNN', batch_size)
    # for m in [m1,m2,m3,m4]:
    #     # with HiddenPrints():
    #     pred = m.predict(dataX, dataY, seq_len=100, pred_len=10)
    #     print(pred)
    #     m.vis_performance(True)
    
    # m = TCN_model('step-TCN', batch_size)
    # m = Seq2Seq_model('step-Seq2Seq', batch_size)
    # m = Seq2SeqPlus_model('step-Seq2SeqPlus', batch_size)
    m = Seq2SeqAtt_model('step-Seq2SeqAtt', batch_size)
    pred = m.predict(dataX, dataY, seq_len=100, pred_len=10)
    m.vis_performance(True)

    return


# directly predict with the single framework, no sliding window, train_test_split: 0.8, 0.2
def pred_nowin_single(seq_len, method, pred_len=10, vis=False):

    trail_name = "sg_nowin_seq%d_%s" %(seq_len, method)
    print(colored(trail_name, 'blue'))

    method_dict = { 'seq2seq+': Seq2SeqPlus_model, 'seq2seq': Seq2Seq_model, 'tcn': TCN_model, 'gru': GRU_model, 'lstm': LSTM_model, 'bpnn': BPNN_model }
    if method not in method_dict.keys():
        print('unrecognized method: ' + method)
        return

    # load data
    df = pd.read_excel('data\df.xlsx', sheet_name='Sheet1')
    dataY = np.array(df['Cprice']).reshape(-1,1)
    dataX = np.array(df.iloc[:,1:]) # carbon price at the previous time step is also used as feature

    trainX = dataX[:int(len(dataX)*0.8), :]
    trainY = dataY[:int(len(dataY)*0.8), :]
    testX = dataX[int(len(dataX)*0.8):, :]
    testY = dataY[int(len(dataY)*0.8):, :]

    m = method_dict[method](trail_name, batch_size)

    # model_path = os.path.join('saved_model', trail_name, 'best_model.pth')
    # if ~os.path.exists(model_path):
        
    # else:
    #     m.load_model(trail_name)

    # # train
    # print('training...')
    # m.prepare_data(trainX, trainY, seq_len, pred_len)
    # m.init_model()
    # m.train_model()
    
    # test
    print('testing...')
    m.prepare_data(dataX, dataY, seq_len, pred_len)
    m.init_model()
    m.load_model(trail_name)
    pred, real = m.apply_model(vis)
    

    # # store
    # f = open(trail_name+".pkl", "wb")
    # pickle.dump((params, pred, real), f)
    # f.close()

    # # # load
    # # f = open("results\\"+trail_name+".pkl", "rb")
    # # params, pred, real = pickle.load(f)
    # # f.close()

    # # print out performance
    # show_performance(trail_name, pred, real, vis)

    return


# directly predict with the single framework, no sliding window, train_test_split: 0.8, 0.2
def pred_nowin_hybrid(seq_len, method, pred_len=10, vis=False):

    trail_name = "sg_nowin_seq%d_%s" %(seq_len, method)
    print(colored(trail_name, 'blue'))

    method_dict = { 'seq2seq+': Seq2SeqPlus_model, 'seq2seq': Seq2Seq_model, 'tcn': TCN_model, 'gru': GRU_model, 'lstm': LSTM_model, 'bpnn': BPNN_model }
    if method not in method_dict.keys():
        print('unrecognized method: ' + method)
        return

    # load data
    df = pd.read_excel('data\df.xlsx', sheet_name='Sheet1')
    dataY = np.array(df['Cprice']).reshape(-1,1)
    dataX = np.array(df.iloc[:,1:]) # carbon price at the previous time step is also used as feature

    trainX = dataX[:int(len(dataX)*0.8), :]
    trainY = dataY[:int(len(dataY)*0.8), :]
    testX = dataX[int(len(dataX)*0.8):, :]
    testY = dataY[int(len(dataY)*0.8):, :]

    m = method_dict[method](trail_name, batch_size)

    # model_path = os.path.join('saved_model', trail_name, 'best_model.pth')
    # if ~os.path.exists(model_path):
        
    # else:
    #     m.load_model(trail_name)

    # train
    print('training...')
    m.prepare_data(trainX, trainY, seq_len, pred_len)
    m.init_model()
    m.train_model()
    
    # test
    print('testing...')
    m.prepare_data(dataX, dataY, seq_len, pred_len)
    m.init_model()
    m.load_model(trail_name)
    pred, real = m.apply_model(vis)
    

    # # store
    # f = open(trail_name+".pkl", "wb")
    # pickle.dump((params, pred, real), f)
    # f.close()

    # # # load
    # # f = open("results\\"+trail_name+".pkl", "rb")
    # # params, pred, real = pickle.load(f)
    # # f.close()

    # # print out performance
    # show_performance(trail_name, pred, real, vis)

    return



if __name__ == '__main__':
    
    # pred_step(win_len=500, t=200)

    # win_len = [1000, 500]
    # seq_len = [200, 100]
    # methods = ['tcn', 'gru', 'lstm', 'bpnn']

    # pred_single(win_len=500, seq_len=100, method='seq2seq+', vis=True, val_num=10)

    # for i in ['seq2seq', 'tcn', 'gru', 'lstm', 'bpnn']:
    #     pred_single(win_len=500, seq_len=100, method=i, vis=True)


    # pred_nowin_single(100, 'seq2seqplus', pred_len=10, vis=True)
    
    from sarimax import forecast_arima
    
    