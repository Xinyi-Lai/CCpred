'''
Functions for series restructuring, including decomposition and integration
'''

import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import pwlf # piecewise linear fit
from termcolor import colored

# entropy helper to indicate the complexity of a series
import EntropyHub as EH
def FuzzyEn2(s:np.ndarray, r=0.2, m=2, n=2):
    th = r * np.std(s)
    return EH.FuzzEn(s, 2, r=(th, n))[0][-1]


############ sequence extension ############
def extend_symmetry(series, ex=5):
    """ extend the series with symmetry to mitigate the windowing effect
        Args:
            series (np.array): the original series, shape of (win_len,)
            ex (int, optional): the number of points to extend, default to be 5.
        Returns:
            series (np.array): the extended series, shape of (win_len+2*ex,)
        Notes:
            after decomposition, truncate the padded part with imfs[:,ex:-ex]
    """
    endpt = series[-1]
    tail = series[-ex-1:-1]
    tail_symm = 2*endpt - tail[::-1]
    startpt = series[0]
    head = series[1:ex+1]
    head_symm = 2*startpt - head[::-1]
    return np.concatenate((head_symm, series, tail_symm))


############ decomposition ############


### EMD++

# CEEMDAN (Complete ensemble EMD with adaptive noise)
def decomp_ceemdan(series, ex=0):
    """decompose the series with CEEMDAN
        Args:
            series (np.array): time series to be decomposed, shape of (win_len,)
            ex (int, optional): the number of points to extend, 0 means no extension.
        Returns:
            series (np.array): the decomposed series, shape of (win_len,)
    """
    if ex != 0:
        series = extend_symmetry(series, ex)
    from PyEMD import CEEMDAN
    ceemdan = CEEMDAN()
    ceemdan.ceemdan(series)
    imfs, res = ceemdan.get_imfs_and_residue()
    if ex != 0:
        imfs = imfs[:,ex:-ex]
    return imfs

# EEMD (Ensemble Empirical Mode Decomposition)
def decomp_eemd(series, ex=0):
    """decompose the series with EEMD
        Args:
            series (np.array): time series to be decomposed, shape of (win_len,)
            ex (int, optional): the number of points to extend, 0 means no extension.
        Returns:
            series (np.array): the decomposed series, shape of (win_len,)
    """
    if ex != 0:
        series = extend_symmetry(series, ex)
    from PyEMD import EEMD
    eemd = EEMD()
    eemd.eemd(series)
    imfs, res = eemd.get_imfs_and_residue()
    imfs = np.concatenate((imfs, res.reshape(1,-1)))
    if ex != 0:
        imfs = imfs[:,ex:-ex]
    return imfs

# EMD (Empirical Mode Decomposition)
def decomp_emd(series, ex=0):
    """decompose the series with EMD
        Args:
            series (np.array): time series to be decomposed, shape of (win_len,)
            ex (int, optional): the number of points to extend, 0 means no extension.
        Returns:
            series (np.array): the decomposed series, shape of (win_len,)
    """
    if ex != 0:
        series = extend_symmetry(series, ex)
    from PyEMD import EMD
    emd = EMD()
    emd.emd(series)
    imfs, res = emd.get_imfs_and_residue()
    imfs = np.concatenate((imfs, res.reshape(1,-1)))
    if ex != 0:
        imfs = imfs[:,ex:-ex]
    return imfs



### EWT (Empirical Wavelet Transform) TODO: to be explored: EWT, ESMD, VMD, DoubleDecomp
def decomp_ewt(series, num_comp):
    import ewtpy
    ewt, mfb, boundaries = ewtpy.EWT1D(series, N=num_comp)
    return ewt.T



############ integration ############

def integr_fuzzen_threshold(imfs, th=0.01):
    """ group imfs based on fuzzy entropy, according to a threshold
        Args:
            imfs (np.array):
            th (int, optional): needs to be carefully determined, default to be 0.01.
        Returns:
            reconstr (np.array): the reconstructed subsequences, [hi-freq, lo-freq], shape of (2, win_len)
    """
    fuzzyEns = np.array([ FuzzyEn2(imfs[i]) for i in range(imfs.shape[0]) ])
    reconstr = np.zeros((2, imfs.shape[1]))
    for i in range(imfs.shape[0]):
        if fuzzyEns[i]>th:  reconstr[0,:] += imfs[i]
        else:               reconstr[1,:] += imfs[i]
    return reconstr


def integr_fuzzen_pwlf(imfs, n_integr=2):
    """ group imfs based on fuzzy entropy, determine threshold(s) by piecewise linear fit
        Args:
            imfs (np.array):
            n_integr (int, optional): number of groups, default to be 2 (hi-freq, lo-freq).
        Returns:
            reconstr (np.array): the reconstructed subsequences, higher fuzzen (hi-freq) comes first, shape of (n_integr, win_len)
    """

    # sort by fuzzy entropy
    fuzzyEns = np.array([ FuzzyEn2(imfs[i]) for i in range(imfs.shape[0]) ])
    sortIdx = fuzzyEns.argsort()[::-1] # higher fuzzen (hi-freq) comes first
    fuzzyEns = fuzzyEns[sortIdx]

    # find change pt with pwlf
    x = range(len(fuzzyEns))
    my_pwlf = pwlf.PiecewiseLinFit(x, fuzzyEns) # piecewise linear fit model
    res = my_pwlf.fit(n_integr)

    # group subsequences
    reconstr = np.zeros((n_integr, imfs.shape[1]))
    for i in range(imfs.shape[0]):
        for j in range(len(res)-1):
            if i >= res[j] and i < res[j+1]+1e-6: # add a tol to include the last imf
                reconstr[j,:] += imfs[i]
                # print(i,j)
                break
    
    # # plot piecewise linear fit result
    # plt.figure(figsize=(12,4))
    # xHat = np.linspace(min(x), max(x), num=10000)
    # yHat = my_pwlf.predict(xHat)
    # plt.stem(fuzzyEns)
    # plt.plot(x, fuzzyEns, '.', label='FuzzyEn scatter')
    # plt.plot(xHat, yHat, '-', label='piecewise linear fit')
    # for i in range(1,len(res)-1):
    #     plt.axvline(x=res[i], c='g', linestyle='--')
    # plt.legend()

    return reconstr


def integr_fine_to_coarse(imfs):
    ''' cumulatively sum imfs until it becomes non-stationary, the stationary parts are hi-freq, the others are lo-freq
        Args:
            imfs (np.array): 
        Returns:
            reconstr (np.array): the reconstructed subsequences, higher fuzzen (hi-freq) comes first, shape of (2, win_len)
    '''
    # sort by fuzzy entropy
    fuzzyEns = np.array([ FuzzyEn2(imfs[i]) for i in range(imfs.shape[0]) ])
    sortIdx = fuzzyEns.argsort()[::-1] # higher fuzzen (hi-freq) comes first
    imfs = imfs[sortIdx]

    from statsmodels.tsa import stattools
    arr = np.zeros(imfs[0].shape)
    for i in range(len(imfs)):
        arr += imfs[i]
        if stattools.adfuller(arr)[1] > 0.001:
            break
    reconstr = np.zeros((2, imfs.shape[1]))
    reconstr[0,:] = np.sum([ imfs[j] for j in range(i)], axis=0)
    reconstr[1,:] = np.sum([ imfs[j] for j in range(i,len(imfs))], axis=0)
    return reconstr

    

############ Singular Spectrum Analysis ############

def restr_ssa(series, ex=0, n_decomp=10, n_integr=5, vis=False):
    """ restructure the series based on sigular spectrum analysis, first decompose and then group based on weighted correlation
        Args:
            series (np.array): time series to be restructure
            ex (int, optional): number of points to extend, 0 means no extension
            n_decomp (int, optional): number of subsequences to be decomposed at first, (window length in SSA), default to be 10
            n_integr (int, optional): number of clusters, default to be 3 (hi-freq, mi-freq, lo-freq)
            vis (bool, optional): whether to show vis results, default to be False
        Returns:
            reconstr (np.array): the reconstructed subsequences, higher fuzzen (hi-freq) comes first, shape of (n_integr, win_len)
    """

    # extend the series
    if ex != 0:
        series = extend_symmetry(series, ex)

    # decompose with SSA
    from ssa import SSA
    ssa = SSA(series, n_decomp)

    # group with AgglomerativeClustering based on wcorr
    from sklearn.cluster import AgglomerativeClustering
    distance_matrix = 1 / ssa.Wcorr
    cls = AgglomerativeClustering(n_clusters=n_integr, linkage='average', affinity='precomputed')
    cls = cls.fit(distance_matrix)
    reconstr = np.zeros((n_integr, len(series)))
    new_sigma = np.zeros(n_integr)
    for i in range(n_integr):
        reconstr[i,:] = ssa.reconstruct(np.where(cls.labels_==i)[0])
        new_sigma[i] = np.sum(ssa.Sigma[np.where(cls.labels_==i)[0]])

    # sort by new sigma
    sortIdx = new_sigma.argsort()[::-1] # higher sigma (signal instead of noise) comes first
    reconstr = reconstr[sortIdx]

    # trim the extended part
    if ex != 0:
        reconstr = reconstr[:,ex:-ex]

    if vis:
        vis_restr(decomposed=np.array(ssa.components_to_df()).T, grouped=reconstr, orig=series)

    return reconstr



############ Visualization ############

# visualize decomposition and integration results
def vis_restr(decomposed, grouped, orig):
    
    # check if we carelessly omitted some important components
    sum_decomposed = np.sum(decomposed,axis=0)
    sum_grouped = np.sum(grouped,axis=0)
    print( np.sqrt(np.mean( (orig-sum_decomposed)**2 )) )
    print( np.sqrt(np.mean( (orig-sum_grouped)**2 )) )

    plt.figure(figsize=(15,15))
    plt.subplot(211)
    for i in range(decomposed.shape[0]):
        plt.plot(decomposed[i], label='sub%d'%i)
    plt.plot(orig, label='orig')
    plt.title('Decomposed')
    plt.legend()

    plt.subplot(212)
    for i in range(grouped.shape[0]):
        plt.plot(grouped[i,:], label='group%d'%i)
    plt.plot(orig, label='y')
    plt.title('Grouped')
    plt.legend()

    plt.show()
    return




############ demo ############
def demo_series_restr(win_len=200, t=500):

    # load data
    df = pd.read_excel('data\source\CCprice.xlsx', sheet_name='Sheet1')
    Cprice = np.array(df['C_Price'])[-1500:]
    # windowing and for one timestep
    win_x = Cprice[t:t+win_len]

    ### decompose
    imfs = decomp_ceemdan(win_x)
    # imfs = decomp_eemd(win_x)
    # imfs = decomp_emd(win_x)
    ### integrate
    reconstr = integr_fuzzen_pwlf(imfs, n_integr=2)
    # reconstr = integr_fine_to_coarse(imfs)
    # reconstr = integr_fuzzen_threshold(imfs)
    ### vis
    vis_restr(imfs, reconstr, win_x)

    # ### ssa
    # reconstr = restr_ssa(win_x, vis=True)

    return




############ prepare data ############
def preparedata_win_restr(win_len:int, method:str, pred_len=10):
    ''' It takes a long time to decompose the windowed data (~5s with CEEMDAN, 1/3 of the iteration time),
        and CEEMDAN/EEMD introduce randomness during decomposition every time we run a simulation.
        So, to accelerate simulation, this script precalculates and stores the decomposition results.
    '''
    import pickle
    from tqdm import tqdm

    trail_name = "restr_win%d_%s" %(win_len, method)
    print(colored(trail_name, 'blue'))

    # load data
    df = pd.read_excel('data\source\CCprice.xlsx', sheet_name='Sheet1')
    Cprice = np.array(df['C_Price'])

    # slide the window, decompose at each step, only for the last 100 steps
    val_num = 100
    win_ys = np.zeros((val_num, pred_len))
    win_restr = []

    for i in tqdm(range(val_num)):
        t = len(Cprice) - val_num - win_len - pred_len + i
        win_x = Cprice[t : t+win_len]
        win_ys[i,:] = Cprice[t+win_len : t+win_len+pred_len]

        if method == 'ceemdan':
            imfs = decomp_ceemdan(win_x)
            reconstr = integr_fuzzen_pwlf(imfs, n_integr=2)
        elif method == 'ceemdan_ex':
            imfs = decomp_ceemdan(win_x, ex=5)
            reconstr = integr_fuzzen_pwlf(imfs, n_integr=2)
        elif method == 'eemd':
            imfs = decomp_eemd(win_x)
            reconstr = integr_fuzzen_pwlf(imfs, n_integr=2)
        elif method == 'emd':
            imfs = decomp_emd(win_x)
            reconstr = integr_fuzzen_pwlf(imfs, n_integr=2)
        # elif method == 'ewt': 
        #     imfs = decomp_ewt(win_x, num_comp=10)
        #     reconstr = integr_fuzzen_pwlf(imfs, n_integr=2)
        elif method == 'ssa':
            reconstr = restr_ssa(win_x)
        elif method == 'ssa_ex':
            reconstr = restr_ssa(win_x, ex=5)
        else:
            print('unrecognized method: ' + method)
            return

        win_restr.append(reconstr)

    # store
    win_restr = np.array(win_restr)
    params = { 'win_len': win_len, 'method': method }
    f = open(trail_name+".pkl", "wb")
    pickle.dump((params, win_restr, win_ys), f)
    f.close()

    # # load
    # f = open(trail_name+".pkl", "rb")
    # params, win_restr, win_ys = pickle.load(f)
    # f.close()

    return



if __name__ == '__main__':

    # demo_series_restr(win_len=200, t=800)

    # methods = ['ssa_ex', 'ssa', 'ceemdan', 'eemd', 'emd']
    # for method in methods:
    #     preparedata_win_restr(win_len=200, win_step=1, method=method)

    for win_len in [1000,500]:
        for method in ['ceemdan_ex', 'ssa_ex']:
            preparedata_win_restr(win_len, method)

