### Data visualization with interactive matplotlib widgets

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib.widgets import Slider, Button, RadioButtons

import pickle
from utils import *


def plot_sg_result(trail_name): #TODO:

    f = open(trail_name+'.pkl', 'rb')
    _, pred, real = pickle.load(f)
    f.close()
    # if trail_name[0:2] == 'hb':
    #     pred = np.sum(pred, axis=0)
    plt.figure()
    plt.title('%s, RMSE=%.2f, MAPE=%.2f%%' %(trail_name, cal_rmse(real,pred), cal_mape(real,pred)))
    plt.plot(pred, label='pred')
    plt.plot(real, label='real')
    plt.legend()
    plt.show()

    return

def plot_result(trail_name):
    f = open(trail_name+'.pkl', 'rb')
    _, pred, real = pickle.load(f)
    f.close()
    if trail_name[0:2] == 'hb':
        pred = np.sum(pred, axis=0)
    plt.figure()
    plt.title('%s, RMSE=%.2f, MAPE=%.2f%%' %(trail_name, cal_rmse(real,pred), cal_mape(real,pred)))
    plt.plot(pred, label='pred')
    plt.plot(real, label='real')
    plt.legend()
    plt.show()
    return



def inspect_results():

    result_list = [
        'hb_win200_sam1_ceemdan_arima_tcn',
        'hb_win200_sam1_ceemdan_arima_gru',
        'hb_win200_sam1_ssa_arima_tcn',
        'hb_win200_sam1_ssa_arima_gru',
        'sg_win200_sam1_tcn',
        'sg_win200_sam1_gru',
    ]

    # load results from file and store in a dict
    data = {}
    f = open('results\\'+result_list[0]+'.pkl', 'rb')
    _, _, real = pickle.load(f)
    f.close()
    data['real'] = real
    for res in result_list:
        f = open('results\\'+res+'.pkl', 'rb')
        _, pred, _ = pickle.load(f)
        f.close()
        if res[0:2] == 'hb':
            pred = np.sum(pred, axis=0)
        data[res] = pred


    # initialize variables
    plot_start = 0
    plot_end = plot_start+200
    # plot_end = len(real)-1


    # The plotting function to be called whenever updated
    def _plot(axes):
        
        ax1 = axes[0]
        ax1.cla()
        
        y_true = real[plot_start:plot_end]
        ax1.plot(y_true, label='true')

        rmse_arr = []
        mape_arr = []
        title_arr = []

        for res in result_list:
            y_pred = data[res][plot_start:plot_end]
            rmse = cal_rmse(y_true,y_pred)
            mape = cal_mape(y_true, y_pred)
            if res[0:2] == 'hb':
                title = res.split('_')[-3] + '-' + res.split('_')[-1]
            else:
                title = res.split('_')[-1]
            rmse_arr.append(rmse)
            mape_arr.append(mape)
            title_arr.append(title)
            
            ax1.plot(y_pred, label='%s, rmse=%.3f, mape=%.3f%%'%(title, rmse, mape))

        ax1.legend(loc=2)


        ax2 = axes[1]
        ax2.cla()
        x = range(len(result_list))
        ax2.bar(x, rmse_arr, 0.35)
        ax2.set_xticks(x)
        ax2.set_xticklabels(title_arr, rotation=50)
        ax2.set_title('rmse')

        ax3 = axes[2]
        ax3.cla()
        x = range(len(result_list))
        ax3.bar(x, mape_arr, 0.35)
        ax3.set_xticks(x)
        ax3.set_xticklabels(title_arr, rotation=50)
        ax3.set_title('mape')


        fig.canvas.draw_idle()
        return


    # Create figure
    fig, axes = plt.subplots(1,3)  
    plt.subplots_adjust(bottom=0.2) # adjust subplot position
    _plot(axes)

    # Configure sliders
    axcolor = 'lightgoldenrodyellow' # slider's color
    axPlotStart = plt.axes([0.25, 0.10, 0.65, 0.03], facecolor=axcolor) # slider's position
    sPlotStart = Slider(axPlotStart, 'axPlotStart ', 0, len(real)-1, valinit=0, valstep=1)  # generate slider
    axcolor = 'lightgoldenrodyellow' # slider's color
    axPlotEnd = plt.axes([0.25, 0.05, 0.65, 0.03], facecolor=axcolor) # slider's position
    sPlotEnd = Slider(axPlotEnd, 'axPlotEnd ', 0, len(real)-1, valinit=len(real)-1, valstep=1)  # generate slider

    def PlotStartFunc(val):
        nonlocal plot_start,plot_end
        plot_start = sPlotStart.val
        plot_end = plot_start + 200
        _plot(axes)
        return
    def PlotEndFunc(val):
        nonlocal plot_end
        plot_end = sPlotEnd.val
        _plot(axes)
        return
        
    sPlotStart.on_changed(PlotStartFunc)
    sPlotEnd.on_changed(PlotEndFunc)

    plt.show()

    return


if __name__ == "__main__":
    
    # inspect_results()

    # plot_result(trail_name='hb_win200_sam1_emd_arima_tcn')

    plot_sg_result(trail_name='sg_win200_sam1_tcn')