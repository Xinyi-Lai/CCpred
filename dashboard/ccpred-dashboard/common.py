import pandas as pd
from sqlalchemy import create_engine


#################### Global variables ####################

# load carbon dataset
disk_engine = create_engine('sqlite:///assets/CarbonPrice.db')
df_chn = pd.read_sql_query('SELECT * FROM df_chn', disk_engine)
df_chn.Date = pd.DatetimeIndex(df_chn.Date).strftime('%Y-%m-%d')
# print(df_chn.columns)
city_list = ['Guangzhou', 'Hubei', 'Shanghai', 'Beijing', 'Fujian', 'Chongqing', 'Tianjin', 'Shenzhen']
Xvar_list = ['EU-CC', 'WTI-Oil', 'Brent-Oil', 'Zhengzhou-Coal', 'Dalian-Coal', 'Rtd-Coal', 'US-NatGas', 'SH-FOil', 'US-FOil', 'CSI300', 'US-DJI', 'USD-CNY']
