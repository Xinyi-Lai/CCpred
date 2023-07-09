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