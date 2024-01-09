import dash
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc

import plotly.graph_objects as go
import pandas as pd
from sqlalchemy import create_engine


dash.register_page(__name__)


################################################################################## 
# load dataset
disk_engine = create_engine('sqlite:///assets/CarbonPrice.db')
df_chn = pd.read_sql_query('SELECT * FROM df_chn', disk_engine)
city_list = ['Guangzhou', 'Hubei', 'Shanghai', 'Beijing', 'Fujian', 'Chongqing', 'Tianjin', 'Shenzhen']
Xvar_list = ['EU-CC', 'WTI-Oil', 'Brent-Oil', 'Zhengzhou-Coal', 'Dalian-Coal', 'Rtd-Coal', 'US-NatGas', 'SH-FOil', 'US-FOil', 'CSI300', 'US-DJI', 'USD-CNY']


layout = dbc.Container(
    [
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader('Carbon Prices', className='text-center'),
                    dbc.CardBody([
                        dcc.Dropdown(id='select-market', multi=True, options=city_list, value=['Guangzhou', 'Hubei', 'Shanghai'], placeholder='Select market...'),
                        dcc.Graph(id='graph-Cprice', className='pb-4'),
                    ]),
                ], className='mt-4'),
            ], width=12, lg=6, className='pt-4'),

            dbc.Col([
                dbc.Card([
                    dbc.CardHeader('Exogenous Variables', className='text-center'),
                    dbc.CardBody([
                        dcc.Dropdown(id='select-Xvar', multi=True, options=Xvar_list, value=['Brent-Oil', 'Rtd-Coal', 'USD-CNY'], placeholder='Select X variable...'),
                    dcc.Graph(id='graph-Xvar', className='pb-4'),
                    ]),
                ], className='mt-4'),
            ], width=12, lg=6, className='pt-4'),
            
        ]),

        dbc.Row([
            dbc.Col(width=4, lg=2, className='pt-4', children=[
                html.H6('Select X vars:'),
                dcc.Checklist(id='select-XvarCorr', options=Xvar_list, value=Xvar_list),
            ]),
            dbc.Col(width=4, lg=2, className='pt-4', children=[
                html.H6('Select Y vars:'),
                dcc.Checklist(id='select-YvarCorr', options=city_list, value=['Guangzhou', 'Hubei', 'Shanghai']),
            ]),
            dbc.Col(width=4, lg=2, className='pt-4', children=[
                html.H6('Select Corr method:'),
                dcc.RadioItems(id='select-CorrType', options=['pearson', 'kendall', 'spearman'], value='pearson'),
            ]),
            
            # dcc.Loading(id='loading-VarCorr', type='circle', children=[
                
            # ]),
            dbc.Col(width=12, lg=6, className='pt-4', children=[
                    dcc.Graph(id='graph-VarCorr', className='pb-4')
                ])
            
        ]),

    ], fluid=True,
)


@callback(
    Output('graph-Cprice', 'figure'),
    Input('select-market', 'value'),
)
def graph_Cprice(selected_markets):
    fig = go.Figure()
    for market in selected_markets:
        fig.add_trace(
            go.Scatter(
                x = df_chn['Date'],
                y = df_chn[market],
                name=market,
                # marker_color='black',
                # line=dict(width=6, dash='dot'),
            )
        )
    fig.update_layout(
        # title='Carbon Prices',
        template='none',
        showlegend=True,
        legend=dict(x=0.01, y=0.99),
        height=400,
        margin=dict(l=40, r=10, t=60, b=55),
        yaxis=dict(title='Price', tickprefix='￥'),
        xaxis=dict(title='Date'),
    )
    return fig

@callback(
    Output('graph-Xvar', 'figure'),
    Input('select-Xvar', 'value'),
)
def graph_Xvar(selected_Xvars):
    fig = go.Figure()
    for x in selected_Xvars:
        fig.add_trace(
            go.Scatter(
                x = df_chn['Date'],
                y = (df_chn[x] - df_chn[x].mean()) / df_chn[x].std(),
                name=x,
            )
        )
    fig.update_layout(
        # title='Exogenous Variables',
        template='none',
        showlegend=True,
        legend=dict(x=0.01, y=0.99),
        height=400,
        margin=dict(l=40, r=10, t=60, b=55),
        yaxis=dict(title='Normalized Value'),
        xaxis=dict(title='Date'),
    )
    return fig

@callback(
    Output('graph-VarCorr', 'figure'),
    Input('select-XvarCorr', 'value'),
    Input('select-YvarCorr', 'value'),
    Input('select-CorrType', 'value'),
)
def graph_VarCorr(selected_XvarCorr, selected_YvarCorr, selected_CorrType):
    df_corr = df_chn[selected_XvarCorr + selected_YvarCorr].corr(method=selected_CorrType)
    fig = go.Figure(data=go.Heatmap(
        z=df_corr.values, 
        x=df_corr.columns, 
        y=df_corr.columns, 
        zmin=-1, zmax=1, colorscale='RdBu_r'
    )) 
    
    fig.update_layout(
        title='Correlation Heatmap',
        template='none',
        showlegend=True,
        legend=dict(x=0.01, y=0.99),
        height=600,
        margin=dict(l=80, r=10, t=60, b=70),
    )
    return fig

