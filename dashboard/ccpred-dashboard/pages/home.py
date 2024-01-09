# -*- coding: utf-8 -*-
import dash
from dash import Dash, dcc, html, dash_table, callback, Input, Output, State, callback_context
import dash_bootstrap_components as dbc

import plotly.graph_objects as go
import pandas as pd
from sqlalchemy import create_engine

dash.register_page(__name__, path='/')

################################################################################## 
# load carbon dataset
disk_engine = create_engine('sqlite:///assets/CarbonPrice.db')
df_chn = pd.read_sql_query('SELECT * FROM df_chn', disk_engine)
# print(df_chn.columns)
city_list = ['Guangzhou', 'Hubei', 'Shanghai', 'Beijing', 'Fujian', 'Chongqing', 'Tianjin', 'Shenzhen']
Xvar_list = ['EU-CC', 'WTI-Oil', 'Brent-Oil', 'Zhengzhou-Coal', 'Dalian-Coal', 'Rtd-Coal', 'US-NatGas', 'SH-FOil', 'US-FOil', 'CSI300', 'US-DJI', 'USD-CNY']


layout = dbc.Container(
    [
        html.H2('Home Page'),
        html.P('This is the home page.'),
        html.P('Project description.'),
        html.P('Data description.'),
    ],
    fluid=True,
)

