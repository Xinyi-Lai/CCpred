import dash
import dash_bootstrap_components as dbc

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.SPACELAB, dbc.icons.FONT_AWESOME],
    use_pages=True,
)

app.layout = dbc.Container(
    [
        dbc.NavbarSimple(
            children=[
                dbc.NavItem(dbc.NavLink('Home', href='/')),
                dbc.NavItem(dbc.NavLink('Correlation', href='correlation')),
                dbc.NavItem(dbc.NavLink('Prediction', href='prediction')),
            ],
            brand='Carbon Price Forecasting',
            brand_href='/',
            color='primary',
            dark=True,
            id='my-top-row',
        ),
        dash.page_container,
    ],
    fluid=True,
)

if __name__ == '__main__':
    app.run(debug=True)
