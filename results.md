# Multi-step prediction results

## single

| win | seq | model | step | RMSE | MAPE |
| --- | --- | ----- | ---- | ---- | ---- |
| 500 | 100 | TCN   | 1    | 2.95 | 2.65%|
| 500 | 100 | TCN   | 2    | 3.60 | 2.65%|
| 500 | 100 | TCN   | 1    | 2.95 | 2.65%|
| 500 | 100 | TCN   | 1    | 2.95 | 2.65%|
| 500 | 100 | TCN   | 1    | 2.95 | 2.65%|
| 500 | 100 | TCN   | 1    | 2.95 | 2.65%|
| 500 | 100 | TCN   | 1    | 2.95 | 2.65%|
| 500 | 100 | TCN   | 1    | 2.95 | 2.65%|
| 500 | 100 | TCN   | 1    | 2.95 | 2.65%|

sg_win500_seq100_tcn
col 0: RMSE=2.95, MAPE=2.65%
col 1: RMSE=3.60, MAPE=3.47%
col 2: RMSE=4.44, MAPE=4.25%
col 3: RMSE=5.23, MAPE=5.02%
col 4: RMSE=6.07, MAPE=5.76%
col 5: RMSE=6.82, MAPE=6.43%
col 6: RMSE=7.40, MAPE=7.13%
col 7: RMSE=7.96, MAPE=7.72%
col 8: RMSE=8.59, MAPE=8.59%
col 9: RMSE=9.04, MAPE=9.08%


sg_win500_seq100_gru
col 0: RMSE=3.27, MAPE=2.93%
col 1: RMSE=4.25, MAPE=4.05%
col 2: RMSE=5.55, MAPE=5.49%
col 3: RMSE=7.03, MAPE=6.60%
col 4: RMSE=8.05, MAPE=7.61%
col 5: RMSE=8.92, MAPE=8.49%
col 6: RMSE=9.72, MAPE=8.99%
col 7: RMSE=10.30, MAPE=9.35%
col 8: RMSE=10.80, MAPE=9.83%
col 9: RMSE=11.20, MAPE=10.20%


sg_win500_seq100_lstm
col 0: RMSE=4.26, MAPE=3.65%
col 1: RMSE=6.01, MAPE=5.20%
col 2: RMSE=7.18, MAPE=6.32%
col 3: RMSE=8.27, MAPE=7.40%
col 4: RMSE=9.03, MAPE=8.19%
col 5: RMSE=9.28, MAPE=8.47%
col 6: RMSE=9.54, MAPE=8.79%
col 7: RMSE=9.89, MAPE=9.02%
col 8: RMSE=10.07, MAPE=9.21%
col 9: RMSE=10.50, MAPE=9.79%

sg_win500_seq100_bpnn
col 0: RMSE=7.82, MAPE=7.37%
col 1: RMSE=8.17, MAPE=7.98%
col 2: RMSE=8.64, MAPE=8.80%
col 3: RMSE=9.37, MAPE=9.37%
col 4: RMSE=9.82, MAPE=9.89%
col 5: RMSE=10.27, MAPE=10.28%
col 6: RMSE=10.82, MAPE=10.84%
col 7: RMSE=10.93, MAPE=11.15%
col 8: RMSE=11.12, MAPE=11.20%
col 9: RMSE=11.06, MAPE=11.26%

window_len = 200

## Lag-1

|step |Method | RMSE | MAPE |
| --- | ----- | ---- | ---- |
| 1   | Lag-1 | 1.23 | 2.35 |


## Single

|step |Method | RMSE | MAPE |
| --- | ----- | ---- | ---- |
| 1   | TCN   | 1.27 | 2.47 |
| 1   | GRU   | 1.36 | 2.85 |
| 1   | ARIMA | 1.42 | 2.50 |
| 1   | BPNN  | 1.39 | 2.92 |
| 1   | LSTM  | 1.46 | 3.00 |
--
| 20  | TCN   | 1.38 | 2.66 |
--
| 50  | TCN   | 1.60 | 2.15 |
| 50  | GRU   | 1.66 | 2.28 |
| 50  | ARIMA | 1.68 | 2.20 |
| 50  | BPNN  | 1.70 | 3.95 |
| 50  | LSTM  | 1.76 | 2.32 |
--
| 100 | TCN   | 2.03 | 2.34 |
| 100 | GRU   | 2.18 | 2.69 |


<br>
<br>


## Hybrid

|step |Decomp   |Pred   | RMSE | MAPE |
| --- | -----   | ----- | ---- | ---- |
| 1   | EMD     | TCN   | 1.59 | 2.92 |
| 1   | EMD     | GRU   | 1.78 | 2.98 |
| 1   | CEEMDAN | TCN   | 1.48 | 2.77 |
| 1   | CEEMDAN | GRU   | 1.47 | 2.76 |
| 1   | SSA     | TCN   | 1.60 | 2.87 |
| 1   | SSA     | GRU   | 1.69 | 2.92 |
--
| 20  | CEEMDAN | TCN   | 1.36 | 2.96 |
| 20  | SSA     | TCN   | 1.64 | 3.70 |
--
| 50  | CEEMDAN | TCN   | 1.64 | 2.72 |
| 50  | CEEMDAN | GRU   | 1.51 | 2.44 |
| 50  | SSA     | TCN   | 1.55 | 2.67 |
| 50  | SSA     | GRU   | 1.51 | 2.74 |
--
| 100 | CEEMDAN | TCN   | 2.05 | 2.88 |
| 100 | CEEMDAN | GRU   | 1.98 | 2.76 |
| 100 | SSA     | TCN   | 2.00 | 2.90 |
| 100 | SSA     | GRU   | 1.88 | 3.10 |
| 100 | SSA_EX  | TCN   | 2.22 | 2.52 |
| 100 | SSA_EX  | GRU   | 2.22 | 2.45 |



last 1000 samples

|step | Decomp  |Pred   | RMSE | MAPE |
| --- | -----   | ----- | ---- | ---- |
| 1   | ExSSA   | TCN   | 2.14 | 2.95 |
| 1   | SSA     | TCN   | 2.36 | 3.35 |
| 1   |ExCEEMDAN| TCN   | 3.41 | 4.53 |
| 1   | CEEMDAN | TCN   | 2.19 | 2.91 |
| 1   |    \    | TCN   | 1.75 | 2.44 |
| 100 | ExSSA-en| TCN   | 1.41 | 3.19 |
| 100 | SSA-en  | TCN   | 2.46 | 3.14 |
| 50  | ExSSA-en| TCN   | 1.50 | 2.99 |
| 50  | SSA-en  | TCN   | 3.07 | 3.22 |



# 0716

sg_win500_seq100_tcn
col 0: RMSE=5.10, MAPE=4.91%    col 1: RMSE=4.95, MAPE=4.77%    col 2: RMSE=5.28, MAPE=4.96%    col 3: RMSE=6.09, MAPE=5.66%    col 4: RMSE=7.00, MAPE=6.51%    col 5: RMSE=7.80, MAPE=6.98%    col 6: RMSE=8.72, MAPE=7.75%    col 7: RMSE=9.51, MAPE=8.29%    col 8: RMSE=10.11, MAPE=8.79%   col 9: RMSE=10.40, MAPE=8.79%

sg_win500_seq100_gru
col 0: RMSE=6.23, MAPE=6.14%    col 1: RMSE=7.22, MAPE=6.71%    col 2: RMSE=8.13, MAPE=7.27%    col 3: RMSE=8.85, MAPE=7.80%    col 4: RMSE=9.47, MAPE=8.40%    col 5: RMSE=9.89, MAPE=8.83%    col 6: RMSE=10.29, MAPE=9.23%   col 7: RMSE=10.65, MAPE=9.51%   col 8: RMSE=10.84, MAPE=9.71%   col 9: RMSE=11.19, MAPE=9.89%

sg_win500_seq100_lstm
col 0: RMSE=7.80, MAPE=7.18%    col 1: RMSE=7.95, MAPE=7.40%    col 2: RMSE=8.96, MAPE=8.27%    col 3: RMSE=10.37, MAPE=9.44%   col 4: RMSE=11.47, MAPE=10.45%  col 5: RMSE=12.38, MAPE=11.18%  col 6: RMSE=13.19, MAPE=11.63%  col 7: RMSE=13.51, MAPE=12.11%  col 8: RMSE=13.07, MAPE=11.85%  col 9: RMSE=11.95, MAPE=11.05%

sg_win500_seq100_bpnn
col 0: RMSE=12.29, MAPE=12.13%  col 1: RMSE=12.67, MAPE=12.15%  col 2: RMSE=12.47, MAPE=11.47%  col 3: RMSE=14.67, MAPE=13.29%  col 4: RMSE=16.41, MAPE=14.93%  col 5: RMSE=16.67, MAPE=15.05%  col 6: RMSE=16.46, MAPE=16.38%  col 7: RMSE=15.37, MAPE=15.19%  col 8: RMSE=17.14, MAPE=17.54%  col 9: RMSE=19.57, MAPE=20.41%



hb_win500_ssa_ex_bpnn_tcn
col 0: RMSE=4.85, MAPE=4.19%    col 1: RMSE=4.86, MAPE=4.28%    col 2: RMSE=5.57, MAPE=4.91%    col 3: RMSE=5.04, MAPE=4.49%    col 4: RMSE=5.58, MAPE=5.14%    col 5: RMSE=6.17, MAPE=5.71%    col 6: RMSE=6.60, MAPE=5.79%    col 7: RMSE=7.43, MAPE=5.93%    col 8: RMSE=8.61, MAPE=6.80%    col 9: RMSE=10.38, MAPE=8.57%

hb_win500_ssa_bpnn_tcn
col 0: RMSE=4.75, MAPE=4.44%    col 1: RMSE=4.96, MAPE=4.57%    col 2: RMSE=4.95, MAPE=4.55%    col 3: RMSE=4.76, MAPE=4.28%    col 4: RMSE=6.57, MAPE=5.75%    col 5: RMSE=8.02, MAPE=6.68%    col 6: RMSE=8.48, MAPE=7.26%    col 7: RMSE=9.40, MAPE=7.49%    col 8: RMSE=8.39, MAPE=6.95%    col 9: RMSE=8.41, MAPE=6.99%

hb_win500_ceemdan_ex_bpnn_tcn
col 0: RMSE=7.61, MAPE=6.97%    col 1: RMSE=7.70, MAPE=7.03%    col 2: RMSE=6.96, MAPE=6.05%    col 3: RMSE=6.41, MAPE=5.92%    col 4: RMSE=7.37, MAPE=7.11%    col 5: RMSE=8.21, MAPE=7.83%    col 6: RMSE=8.68, MAPE=8.07%    col 7: RMSE=8.99, MAPE=7.91%    col 8: RMSE=9.51, MAPE=7.94%
col 9: RMSE=9.34, MAPE=7.85%

hb_win500_ceemdan_bpnn_tcn
col 0: RMSE=6.58, MAPE=5.91%    col 1: RMSE=7.38, MAPE=7.07%    col 2: RMSE=7.72, MAPE=7.21%    col 3: RMSE=7.54, MAPE=7.15%    col 4: RMSE=7.40, MAPE=7.06%    col 5: RMSE=7.57, MAPE=6.81%    col 6: RMSE=8.91, MAPE=8.13%    col 7: RMSE=9.20, MAPE=8.49%    col 8: RMSE=8.45, MAPE=7.92%    col 9: RMSE=7.54, MAPE=6.99%




sg_win800_seq100_tcn
col 0: RMSE=2.90, MAPE=2.80%    col 1: RMSE=3.26, MAPE=2.99%    col 2: RMSE=3.55, MAPE=3.36%    col 3: RMSE=4.01, MAPE=3.66%    col 4: RMSE=3.93, MAPE=3.61%    col 5: RMSE=3.87, MAPE=3.65%    col 6: RMSE=4.02, MAPE=3.81%    col 7: RMSE=3.89, MAPE=3.73%    col 8: RMSE=4.13, MAPE=3.95%    col 9: RMSE=4.31, MAPE=4.01%

sg_win800_seq100_gru
col 0: RMSE=4.52, MAPE=4.04%    col 1: RMSE=4.61, MAPE=4.16%    col 2: RMSE=4.94, MAPE=4.40%    col 3: RMSE=5.20, MAPE=4.74%    col 4: RMSE=5.62, MAPE=5.00%    col 5: RMSE=6.00, MAPE=4.92%    col 6: RMSE=6.57, MAPE=5.18%    col 7: RMSE=7.08, MAPE=5.30%    col 8: RMSE=7.29, MAPE=5.15%    col 9: RMSE=7.46, MAPE=5.22%

sg_win800_seq100_lstm
col 0: RMSE=5.08, MAPE=4.78%    col 1: RMSE=5.78, MAPE=5.35%    col 2: RMSE=6.56, MAPE=6.38%    col 3: RMSE=7.59, MAPE=7.21%    col 4: RMSE=8.25, MAPE=7.90%    col 5: RMSE=8.48, MAPE=8.07%    col 6: RMSE=8.44, MAPE=8.09%    col 7: RMSE=8.04, MAPE=7.55%    col 8: RMSE=7.51, MAPE=7.11%    col 9: RMSE=7.23, MAPE=6.81%

sg_win800_seq100_bpnn
col 0: RMSE=25.59, MAPE=21.90%  col 1: RMSE=19.13, MAPE=16.82%  col 2: RMSE=19.08, MAPE=15.77%  col 3: RMSE=21.22, MAPE=17.04%  col 4: RMSE=19.73, MAPE=15.59%  col 5: RMSE=20.12, MAPE=16.28%  col 6: RMSE=17.63, MAPE=14.49%  col 7: RMSE=17.20, MAPE=14.41%  col 8: RMSE=19.52, MAPE=16.24%  col 9: RMSE=22.53, MAPE=18.68% 


hb_win800_ceemdan_ex_bpnn_tcn
col 0: RMSE=11.18, MAPE=8.88%   col 1: RMSE=10.28, MAPE=8.50%   col 2: RMSE=9.20, MAPE=8.06%    col 3: RMSE=11.15, MAPE=9.78%   col 4: RMSE=12.77, MAPE=10.87%  col 5: RMSE=11.10, MAPE=9.30%   col 6: RMSE=10.28, MAPE=9.41%   col 7: RMSE=10.36, MAPE=8.88%   col 8: RMSE=9.67, MAPE=8.30%    col 9: RMSE=13.87, MAPE=11.63%

hb_win800_ceemdan_bpnn_tcn
col 0: RMSE=12.82, MAPE=9.79%   col 1: RMSE=10.50, MAPE=8.44%   col 2: RMSE=9.73, MAPE=7.80%    col 3: RMSE=11.11, MAPE=8.99%   col 4: RMSE=12.03, MAPE=9.93%   col 5: RMSE=15.11, MAPE=12.39%  col 6: RMSE=16.18, MAPE=13.09%  col 7: RMSE=18.04, MAPE=14.13%  col 8: RMSE=21.98, MAPE=16.85%  col 9: RMSE=19.48, MAPE=15.44%

hb_win800_ssa_ex_bpnn_tcn
col 0: RMSE=6.06, MAPE=4.86%    col 1: RMSE=6.07, MAPE=5.01%    col 2: RMSE=6.13, MAPE=5.36%    col 3: RMSE=6.61, MAPE=5.64%    col 4: RMSE=7.02, MAPE=6.08%    col 5: RMSE=7.27, MAPE=6.51%    col 6: RMSE=8.05, MAPE=6.76%    col 7: RMSE=8.66, MAPE=7.10%    col 8: RMSE=9.31, MAPE=7.92%    col 9: RMSE=9.67, MAPE=8.35%

hb_win800_ssa_bpnn_tcn
col 0: RMSE=7.01, MAPE=6.15%    col 1: RMSE=7.32, MAPE=6.36%    col 2: RMSE=7.06, MAPE=6.01%    col 3: RMSE=7.20, MAPE=6.42%    col 4: RMSE=7.29, MAPE=6.65%    col 5: RMSE=7.12, MAPE=6.42%    col 6: RMSE=7.46, MAPE=6.86%    col 7: RMSE=8.28, MAPE=7.13%    col 8: RMSE=8.56, MAPE=7.29%    col 9: RMSE=8.84, MAPE=7.52%