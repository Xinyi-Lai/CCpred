### Data visualization with interactive matplotlib widgets

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib.widgets import Slider, Button, RadioButtons

import pickle
from utils import *


def plot_pred_results():

    f = open('hb_win200_sam1_ceemdan_arima_gru.pkl', 'rb')
    _, hb_ceemdan_gru, real = pickle.load(f)
    f.close()
    f = open('hb_win200_sam1_ceemdan_arima_tcn.pkl', 'rb')
    _, hb_ceemdan_tcn, _ = pickle.load(f)
    f.close()
    f = open('hb_win200_sam1_ssa_arima_gru.pkl', 'rb')
    _, hb_ssa_gru, _ = pickle.load(f)
    f.close()
    f = open('hb_win200_sam1_ssa_arima_tcn.pkl', 'rb')
    _, hb_ssa_tcn, _ = pickle.load(f)
    f.close()

    f = open('results\sg_win200_sam1_arima.pkl', 'rb')
    _, sg_arima, real = pickle.load(f)
    f.close()
    f = open('results\sg_win200_sam1_bpnn.pkl', 'rb')
    _, sg_bpnn, _ = pickle.load(f)
    f.close()
    f = open('results\sg_win200_sam1_gru.pkl', 'rb')
    _, sg_gru, _ = pickle.load(f)
    f.close()
    f = open('results\sg_win200_sam1_lstm.pkl', 'rb')
    _, sg_lstm, _ = pickle.load(f)
    f.close()
    f = open('results\sg_win200_sam1_tcn.pkl', 'rb')
    _, sg_tcn, _ = pickle.load(f)
    f.close()

    
    # initialize variables
    plot_start = 0
    plot_end = len(real)-1


    # The plotting function to be called whenever updated
    def _plot(ax):
        ax.cla()
        y_true = real[plot_start:plot_end]
        ax.plot(y_true, label='true')
        
        y_pred = np.sum(hb_ssa_tcn, axis=0)[plot_start:plot_end]
        ax.plot(y_pred, label='ssa_tcn, rmse=%.3f, mape=%.3f%%'%(rmse(y_true,y_pred), mape(y_true, y_pred)))
        y_pred = np.sum(hb_ssa_gru, axis=0)[plot_start:plot_end]
        ax.plot(y_pred, label='ssa_gru, rmse=%.3f, mape=%.3f%%'%(rmse(y_true,y_pred), mape(y_true, y_pred)))
        y_pred = np.sum(hb_ceemdan_gru, axis=0)[plot_start:plot_end]
        ax.plot(y_pred, label='ceemdan_gru, rmse=%.3f, mape=%.3f%%'%(rmse(y_true,y_pred), mape(y_true, y_pred)))
        y_pred = np.sum(hb_ceemdan_tcn, axis=0)[plot_start:plot_end]
        ax.plot(y_pred, label='ceemdan_tcn, rmse=%.3f, mape=%.3f%%'%(rmse(y_true,y_pred), mape(y_true, y_pred)))

        y_pred = sg_arima[plot_start:plot_end]
        ax.plot(y_pred, label='sg_arima, rmse=%.3f, mape=%.3f%%'%(rmse(y_true,y_pred), mape(y_true, y_pred)))
        y_pred = sg_bpnn[plot_start:plot_end]
        ax.plot(y_pred, label='sg_bpnn, rmse=%.3f, mape=%.3f%%'%(rmse(y_true,y_pred), mape(y_true, y_pred)))
        y_pred = sg_gru[plot_start:plot_end]
        ax.plot(y_pred, label='sg_gru, rmse=%.3f, mape=%.3f%%'%(rmse(y_true,y_pred), mape(y_true, y_pred)))
        y_pred = sg_lstm[plot_start:plot_end]
        ax.plot(y_pred, label='sg_lstm, rmse=%.3f, mape=%.3f%%'%(rmse(y_true,y_pred), mape(y_true, y_pred)))
        y_pred = sg_tcn[plot_start:plot_end]
        ax.plot(y_pred, label='sg_tcn, rmse=%.3f, mape=%.3f%%'%(rmse(y_true,y_pred), mape(y_true, y_pred)))

        ax.legend()
        fig.canvas.draw_idle()
        return

    # Create figure
    fig, ax = plt.subplots()  
    plt.subplots_adjust(bottom=0.2,left=0.3) # adjust subplot position
    _plot(ax)

    # Configure sliders
    axcolor = 'lightgoldenrodyellow' # slider's color
    axPlotStart = plt.axes([0.25, 0.10, 0.65, 0.03], facecolor=axcolor) # slider's position
    sPlotStart = Slider(axPlotStart, 'axPlotStart ', 0, len(real)-1, valinit=0, valstep=1)  # generate slider
    axcolor = 'lightgoldenrodyellow' # slider's color
    axPlotEnd = plt.axes([0.25, 0.05, 0.65, 0.03], facecolor=axcolor) # slider's position
    sPlotEnd = Slider(axPlotEnd, 'axPlotEnd ', 0, len(real)-1, valinit=len(real)-1, valstep=1)  # generate slider

    def PlotStartFunc(val):
        nonlocal plot_start
        plot_start = sPlotStart.val
        _plot(ax)
        return
    def PlotEndFunc(val):
        nonlocal plot_end
        plot_end = sPlotEnd.val
        _plot(ax)
        return
    sPlotStart.on_changed(PlotStartFunc)
    sPlotEnd.on_changed(PlotEndFunc)

    plt.show()

    return


if __name__ == "__main__":
    
    plot_pred_results()