import optuna
from optuna.trial import TrialState

from train import *


def objective(trial):

    params = {
        # data params
        'batch_size':   trial.suggest_categorical('batch_size', [2,4,8,16,32,64]), # 4, # 
        'seq_len':      trial.suggest_int('seq_len', 5,60), # 22, # 
        'prev_step':    trial.suggest_int('prev_step', 5,60), # 8, # 
        # model params
        'cnn_out_channels': trial.suggest_int('cnn_out_channels', 2,8), # 7, # 
        'cnn_kernel_size':  4, # trial.suggest_int('cnn_kernel_size', 2,4), # 
        'lstm_hidden_size': trial.suggest_int('lstm_hidden_size', 2,8), # 6, # 
        'lstm_num_layers':  1, # trial.suggest_int('lstm_num_layers', 1,4), # 
        'input_size': None, # calculated
        # learning params
        'loss_func':    'MSELoss', # trial.suggest_categorical('loss_func', list(loss_funcs.keys())), # 
        'optimizer':    'RMSprop', # trial.suggest_categorical('optimizer', ['Adam', 'RMSprop', 'SGD']), # 
        'lr':           trial.suggest_float('lr', 1e-5, 1e-1, log=True), # 0.0382554
        'weight_decay': 0.000376358, # trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True), # 
    }

    with HiddenPrints():
        res = train_model(params, trial, output=False)

    return res
    


if __name__ == "__main__":
    study = optuna.create_study(study_name="cnn-lstm-05230939", direction="minimize", storage="sqlite:///db.sqlite3", load_if_exists=False)
    # $ optuna-dashboard sqlite:///db.sqlite3
    study.optimize(objective, n_trials=100, timeout=6000)

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


    