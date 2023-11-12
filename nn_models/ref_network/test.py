from train import *

params = params_CNN_LSTM
batch_size = params['batch_size']
seq_len = params['seq_len']
prev_step = params['prev_step']

######### prepare data for testing #########
_, _, Dte, input_feature_num = load_split_data(data_path, train_ratio=0, val_ratio=0)
Dte = process(Dte, seq_len, prev_step, batch_size, False, 'test set')
params['input_size'] = input_feature_num + prev_step

######### load the trained model for testing #########
model_path = os.path.join(model_dir, 'best_model.pth')
print(colored('------------loading model from {}-------------'.format(model_path), 'blue'))
# model = LSTM(input_size, lstm_hidden_size, lstm_num_layers)
# model = CNN2_LSTM2(seq_len, input_size)
model = CNN_LSTM(params)
model.load_state_dict(torch.load(model_path))
print(model)

######################### testing #########################

_ = test_and_show_results(model, Dte, True, 799, 1115)

