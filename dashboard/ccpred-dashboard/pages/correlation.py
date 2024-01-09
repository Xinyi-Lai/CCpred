import dash
from dash import dcc, html, Input, Output, dash_table, callback
import dash_bootstrap_components as dbc

import pandas as pd
import plotly.graph_objects as go
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
        # TODO move to home page, data description
        ### download data
        dbc.ButtonGroup([
            dbc.Button('Download CSV', id='down-btn-csv'),
            dcc.Download(id='down-df-csv'),
            dbc.Button('Download Excel', id='down-btn-excel'), 
            dcc.Download(id='down-df-excel'),
            dbc.Button('Download db', href='/assets/CarbonPrice.db', download='DownCarbonPrice.db', external_link=True),
            dbc.Button('Download Paper', id='down-btn-paper', disabled=True), 
        ], class_name='mt-3 gap-2'),

        ### show Xvars and Yvars
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader('Carbon Prices', class_name='text-center'),
                    dbc.CardBody([
                        dcc.Dropdown(id='select-market', multi=True, options=city_list, value=['Guangzhou', 'Hubei', 'Shanghai'], placeholder='Select market...'),
                        dcc.Graph(id='graph-Cprice'),
                    ]),
                ]),
            ], class_name='mt-4 mb-2'),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader('Exogenous Variables', class_name='text-center'),
                    dbc.CardBody([
                        dcc.Dropdown(id='select-Xvar', multi=True, options=Xvar_list, value=['Brent-Oil', 'Rtd-Coal', 'USD-CNY'], placeholder='Select X variable...'),
                        dcc.Graph(id='graph-Xvar'),
                    ]),
                ]),
            ], class_name='mt-4 mb-2'),
        ], class_name='row-cols-1 row-cols-lg-2'),

        dash_table.DataTable(
            id='table-data',
            columns=[{"name": i, "id": i} for i in df_chn.columns],
            data=df_chn.to_dict('records'),
            style_cell={'textAlign': 'center', 'minWidth': '100px', 'width': '100px', 'maxWidth': '100px'},
            style_header={'fontWeight': 'bold'},
            style_data_conditional=[{'if': {'row_index': 'odd'}, 'backgroundColor': 'rgb(248, 248, 248)'}],
            style_table={'overflowX': 'auto'},
            page_size=20,
            fixed_rows={'headers': True},
            style_as_list_view=True,
            style_cell_conditional=[{'if': {'column_id': 'Date'}, 'width': '150px'}],
            style_data={'whiteSpace': 'normal', 'height': 'auto'},
        ),

        ### show correlation
        dbc.Card([
            dbc.CardHeader('Correlation Analysis', class_name='text-center'),
            dbc.CardBody([
                dbc.Row([
                    # correlation options
                    dbc.Col([
                        dbc.Row([
                            dbc.Col([html.H6('Select X vars:')]),
                            dbc.Col([html.H6('Select Y vars:')]),
                            dbc.Col([html.H6('Select Corr method:')]),
                            dbc.Col([dcc.Checklist(id='select-XvarCorr', options=Xvar_list, value=Xvar_list)]),
                            dbc.Col([dcc.Checklist(id='select-YvarCorr', options=city_list, value=['Guangzhou', 'Hubei', 'Shanghai'])]),
                            dbc.Col([dcc.RadioItems(id='select-CorrType', options=['pearson', 'kendall', 'spearman'], value='pearson')]),
                        ], class_name='row-cols-3') # 3 cols in a row and align automatically
                    ], class_name='mt-4'),
                    # correlation graph
                    dbc.Col([
                        dcc.Loading(id='loading-VarCorr', type='graph', children=[
                            dcc.Graph(id='graph-VarCorr')
                        ]),  
                    ])
                ], class_name='row-cols-1 row-cols-lg-2')
            ], class_name='ms-3'),
        ], class_name='mt-3'),

        ### other correlation analysis
        dbc.Card([
            dbc.CardHeader('Other Correlation Analysis', class_name='text-center'),
            dbc.CardBody([
                dbc.Row( [ dbc.Col([html.H6('Analysis')]) ]*20, class_name='row-cols-4 gap-3') # 3 cols in a row and align automatically
            ]),
        ], class_name='mt-5'),

    ], fluid=True,
)


# FUTURE: upload data
### download data
@callback(
    Output('down-df-csv', 'data'),
    Input('down-btn-csv', 'n_clicks'),
    prevent_initial_call=True,
)
def down_df_csv(n_clicks):
    return dcc.send_data_frame(df_chn.to_csv, "mydf.csv")

@callback(
    Output('down-df-excel', 'data'),
    Input('down-btn-excel', 'n_clicks'),
    prevent_initial_call=True,
)
def down_df_excel(n_clicks):
    return dcc.send_data_frame(df_chn.to_excel, "mydf.xlsx", sheet_name="Sheet_name_1")


### show Xvars and Yvars
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
        margin=dict(l=40, r=10, t=30, b=55),
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
        margin=dict(l=40, r=10, t=30, b=55),
        yaxis=dict(title='Normalized Value'),
        xaxis=dict(title='Date'),
    )
    return fig

### show Correlation
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
        height=500,
        margin=dict(l=80, r=10, t=60, b=70),
    )
    return fig

