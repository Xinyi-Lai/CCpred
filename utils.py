import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# root mean square error
def cal_rmse(y_true, y_pred):
    y_true, y_pred = np.array(y_true).reshape(-1), np.array(y_pred).reshape(-1)
    assert y_true.shape == y_pred.shape, 'shapes do not match'
    return np.sqrt(np.mean(np.square(y_true-y_pred)))

# mean square error
def cal_mse(y_true, y_pred):
    y_true, y_pred = np.array(y_true).reshape(-1), np.array(y_pred).reshape(-1)
    assert y_true.shape == y_pred.shape, 'shapes do not match'
    return np.mean(np.square(y_true-y_pred))

# mean absolute error
def cal_mae(y_true, y_pred):
    y_true, y_pred = np.array(y_true).reshape(-1), np.array(y_pred).reshape(-1)
    assert y_true.shape == y_pred.shape, 'shapes do not match'
    return np.mean(np.abs(y_true-y_pred))

# mean absolute percent error
def cal_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true).reshape(-1), np.array(y_pred).reshape(-1)
    assert y_true.shape == y_pred.shape, 'shapes do not match'
    return np.mean(np.abs((y_true-y_pred)/y_true)) * 100


# use to block out prints when tuning
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


# print metrics (rmse and mape) of each columns
# plot the first, middle, and last columns
def show_performance(trail_name, pred, real, vis):
    print('performance of %s' %trail_name)
    for i in range(pred.shape[1]):
        rmse = cal_rmse(real[:,i], pred[:,i])
        mape = cal_mape(real[:,i], pred[:,i])
        print('col %d: RMSE=%.2f, MAPE=%.2f%%' %(i, rmse, mape))
    if vis:
        # plot the first, middle, and last columns
        f, axes = plt.subplots(1,3)
        f.suptitle('performance of %s' %trail_name)
        for idx, icol in enumerate([0, pred.shape[1]//2, pred.shape[1]-1]):
            ax = axes[idx]
            r = real[:,icol]
            p = pred[:,icol]
            ax.plot(r, label='real')
            ax.plot(p, label='pred')
            ax.set_title('col%d, RMSE=%.2f, MAPE=%.2f%%' %(icol, cal_rmse(r,p), cal_mape(r,p)))
            ax.legend()
        plt.show()