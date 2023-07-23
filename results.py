import pandas as pd
import pickle

from utils import show_performance

trail_name = 'sg_win500_seq100_tcn'

df = pd.DataFrame()

f = open('results\\'+trail_name+'.pkl', 'rb')
_, pred, real = pickle.load(f)
f.close()

df['pred'] = pred
print(df)

# show_performance(trail_name, pred, real, True)
