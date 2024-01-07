import pandas as pd

from sqlalchemy import create_engine

disk_engine = create_engine('sqlite:///CarbonPrice.db')


df = pd.read_excel('data/df_chn.xlsx', sheet_name='df-chn-itpl') 
df.to_sql('df_chn', disk_engine, if_exists='replace', index=False)
print(df.head())

df_chn = pd.read_sql_query('SELECT * FROM df_chn LIMIT 3', disk_engine)
print(df_chn.head())


