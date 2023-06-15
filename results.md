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