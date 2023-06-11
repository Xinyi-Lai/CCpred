# A window-based hybrid forecasting framework for carbon prices

## CCpred pipeline:

1. Decomposition: EMD, EEMD, CEEMDAN, SSA, 

    (to be explored: EWT, VMD, LMD, ESMD, quadratic decomp)

2. Integration: 
    
    a. Cri: FuzzEn, w-corr, (to be explored: SampEn, SingularEn...)
    
    b. Grouping: thresholding, fine-to-coarse, pwlf, AgglomerativeClustering

3. Forecast: 

    a. hi-freq: arima, (to be explored: non-linear autoregressive, elman network)
    
    b. mid-freq: (to be explored: wavelet neural network)

    c. lo-freq: bpnn, lstm, gru, tcn, (to be explored: transformer, informer, diffusion-transformer, support vector machine)

4. Ensemble

5. Error correction: arima, tcn

TODO: Dash (plotly) + sqlite3

---

## Copying a conda environment:

export: `conda env export > environment.yml`

import: `conda env create -f environment.yml`

may change the env name in the first line