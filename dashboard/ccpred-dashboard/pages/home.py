# -*- coding: utf-8 -*-
import dash
from dash import dcc, html, dash_table, callback, Input, Output
from dash.dash_table.Format import Format, Scheme
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

from common import *


dash.register_page(__name__, path='/')



#################### Page components and callbacks ####################


###### Tab2 Data

### card-download-btn
card_data_download = dbc.Card([
    dbc.CardBody([
        dbc.ButtonGroup([
            dbc.Button('Download CSV', id='down-btn-csv', outline=True, color='primary'),
            dcc.Download(id='down-df-csv'),
            dbc.Button('Download Excel', id='down-btn-excel', outline=True, color='primary'), 
            dcc.Download(id='down-df-excel'),
            dbc.Button('Download db', id='down-btn-db', href='/assets/CarbonPrice.db', download='DownCarbonPrice.db', external_link=True, color='secondary'),
            dbc.Tooltip('NOT recommended...', target='down-btn-db', placement='top'),
            dbc.Button('Download Paper', id='down-btn-paper', color='secondary'), 
            dbc.Tooltip('Coming soon...', target='down-btn-paper', placement='top'),
        ], class_name='gap-3'),
    ]),
], color='light', outline=True, class_name='mt-4')

# download-callback-csv
@callback(
    Output('down-df-csv', 'data'),
    Input('down-btn-csv', 'n_clicks'),
    prevent_initial_call=True,
)
def down_df_csv(n_clicks):
    return dcc.send_data_frame(df_chn.to_csv, 'mydf.csv')
# download-callback-excel
@callback(
    Output('down-df-excel', 'data'),
    Input('down-btn-excel', 'n_clicks'),
    prevent_initial_call=True,
)
def down_df_excel(n_clicks):
    return dcc.send_data_frame(df_chn.to_excel, 'mydf.xlsx', sheet_name='Sheet_name_1')


### card-graph
card_data_graph = dbc.Card([
    # dbc.CardHeader('Carbon Prices and Exogenous Variables', class_name='text-center'),
    dbc.CardBody([
        html.H6('Select carbon markets:'),
        dcc.Dropdown(id='dropdown-market', multi=True, options=city_list, value=['Guangzhou', 'Hubei', 'Shanghai'], placeholder='Select market...'),
        html.Br(),
        html.H6('Select explanatory variables:'),
        dcc.Dropdown(id='dropdown-Xvar', multi=True, options=Xvar_list, value=['EU-CC', 'Brent-Oil', 'Rtd-Coal', 'US-DJI'], placeholder='Select Xvar...'),
        dcc.Graph(id='graph-overview'),
    ]),
], class_name='mt-4 w-80')

# card-graph-callback
@callback(
    Output('graph-overview', 'figure'),
    Input('dropdown-market', 'value'),
    Input('dropdown-Xvar', 'value'),
)
def graph_overview(selected_markets, selected_xvars):
    fig = go.Figure()

    for market in selected_markets:
        fig.add_trace(go.Scatter(
            x = df_chn['Date'],
            y = df_chn[market],
            name=market,
            yaxis='y2',
        ))
    for xvar in selected_xvars:
        fig.add_trace(go.Scatter(
            x = df_chn['Date'],
            y = (df_chn[xvar] - df_chn[xvar].mean()) / df_chn[xvar].std(),
            name=xvar,
            yaxis='y1',
        ))

    fig.update_layout(
        template='none',
        height=700,
        margin=dict(l=150, r=150, t=60, b=60),
        # dragmode='pan',
        # legend=dict(x=0.01, y=0.99),
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label='1y', step='year', stepmode='backward'),
                    dict(count=3, label='3y', step='year', stepmode='backward'),
                    dict(step='all'),
                ])
            ),
            rangeslider=dict(visible=True),
            title='Date',
            type='date',
        ),
        # xvars
        yaxis1=dict(
            anchor='x',
            domain=[0, 0.48],
            linecolor='#673ab7',
            mirror=True,
            showline=True,
            side='left',
            tickfont={'color': '#673ab7'},
            title='Normalized Value',
            titlefont={'color': '#673ab7'},
            zeroline=False
        ),
        # prices
        yaxis2=dict(
            anchor='x',
            domain=[0.52, 1],
            linecolor='#2196F3',
            mirror=True,
            showline=True,
            side='left',
            tickfont={'color': '#2196F3'},
            title='Price (CNY/ton)',
            titlefont={'color': '#2196F3'},
            zeroline=True
        ),
        # subtitles
        annotations=[
            dict(
                font=dict(size=16, color='#2196F3'),
                showarrow=False,
                text='Carbon Markets',
                x=0.50, xanchor='center', xref='paper',
                y=1.00, yanchor='bottom', yref='paper',
            ),
            dict(
                font=dict(size=16, color='#673ab7'),
                showarrow=False,
                text='Exogenous Variables',
                x=0.50, xanchor='center', xref='paper',
                y=0.45, yanchor='bottom', yref='paper',
            )
        ],
    )
    # alternatively, pass in a dict to respectively update layout 
    # fig['layout']['yaxis'].update({'title':'Price', 'tickprefix':'￥'})
    # fig['layout']['yaxis2'].update({'title':'Normalized Value'})

    return fig


### card-table
table_columns = [ dict(name='Date', id='Date', type='datetime') ]
table_columns.extend([dict(name=i, id=i, hideable=True, type='numeric', format=Format(precision=3, scheme=Scheme.fixed)) for i in df_chn.columns[2:]])
card_data_table = dbc.Card([
    # dbc.CardHeader('Carbon Prices and Exogenous Variables', class_name='text-center'),
    dbc.CardBody([
        dash_table.DataTable(
            columns=table_columns,
            data=df_chn.iloc[::-1].to_dict('records'),
            fixed_columns={'headers': True, 'data':1},
            fixed_rows={'headers': True},
            page_size=20,
            style_as_list_view=True,
            style_cell={'textAlign': 'center', 'minWidth': '100px', 'width': '100px', 'maxWidth': '100px'},
            style_cell_conditional=[{'if': {'column_id': 'Date'}, 'width': '150px'}],
            style_header={'fontWeight': 'bold'},
            style_data={'whiteSpace': 'normal', 'height': 'auto'},
            style_data_conditional=[{'if': {'row_index': 'odd'}, 'backgroundColor': 'rgb(248, 248, 248)'}],
            style_table={'overflowX': 'auto', 'minWidth': '100%'},
        ),
    ], class_name='ms-5 me-5 mt-3'),
], class_name='mt-4')


#################### Layout ####################

layout = dbc.Container(
    [
        ### Tabs
        dbc.Tabs(
            [
                # Tab1: Project description
                dbc.Tab(
                    [
                        dbc.Card([
                                dbc.CardHeader(f'Description #{i}', class_name='text-center'),
                                dbc.CardBody(' description description description '*50),
                        ], class_name='mt-4') for i in range(3)
                    ], 
                    label='Project description',
                ),

                # Tab2: Data overview
                dbc.Tab([
                    card_data_download,
                    card_data_graph,
                    card_data_table,
                ], label='Data overview'),
                
                # Tab3: Source
                dbc.Tab(
                    [
                        dbc.Card([
                                dbc.CardHeader(f'Description #{i}', class_name='text-center'),
                                dbc.CardBody(' description description description '*50),
                        ], class_name='mt-4') for i in range(3)
                    ], label='Source & References',
                ),
            ],
            active_tab='tab-1',
            class_name='mt-4',
        ),
    ],
    # fluid=True,
)


