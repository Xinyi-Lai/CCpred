import dash
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

from common import *

dash.register_page(__name__)


# TODO
#################### Page components and callbacks ####################



#################### Layout ####################


layout = dbc.Container(
    [
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

    ], 
    # fluid=True,
)



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

