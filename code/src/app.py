import dash
from dash.dependencies import Input, Output
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash import no_update
from data_cache import DataCache

# Import the page components
from pages.page1 import page1_layout, register_callbacks_page1
from pages.page2 import page2_layout, register_callbacks_page2
from pages.page3 import page3_layout, register_callbacks_page3
from pages.page4 import page4_layout, register_callbacks_page4
from pages.page5 import page5_layout, register_callbacks_page5
# Initialize the Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True,
                external_stylesheets=[dbc.themes.BOOTSTRAP, 'assets/style.css'])

# Front Page Layout
front_page_layout = html.Div(
    [
        # Header
        html.Div(
            [
                html.H1("Thesis Dashboard", style={
                        "font-size": "3rem", "margin-bottom": "0.5rem"}),
                html.H4("Directional quality assessment indicators for nonlinear dimensionality reduction and data visualisation",
                        style={"font-weight": "300", "margin-bottom": "2rem"}),
            ],
            style={"text-align": "center", "margin": "3rem 0"}
        ),
        # Cards for Navigation
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardBody(
                                [
                                    html.H5(
                                        "Page 1: Convergence of the algorithm", className="card-title"),
                                    html.P("Convergence of the algorithm",
                                           className="card-text"),
                                    # Wrapping the button in a div for centering
                                    html.Div(
                                        dbc.Button(
                                            "Go to Page 1", color="primary", href="/page1"),
                                        style={"margin-top": "auto",
                                               "text-align": "center"}
                                    ),
                                ],
                                style={
                                    "display": "flex", "flex-direction": "column", "height": "100%"}
                            )
                        ],
                        style={"height": "100%"},
                    ),
                    width=4
                ),
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardBody(
                                [
                                    html.H5("Page 2: Path by path analysis",
                                            className="card-title"),
                                    html.P("Path by path analysis",
                                           className="card-text"),
                                    # Wrapping the button in a div for centering
                                    html.Div(
                                        dbc.Button(
                                            "Go to Page 2", color="primary", href="/page2"),
                                        style={"margin-top": "auto",
                                               "text-align": "center"}
                                    ),
                                ],
                                style={
                                    "display": "flex", "flex-direction": "column", "height": "100%"}
                            )
                        ],
                        style={"height": "100%"},
                    ),
                    width=4
                ),
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardBody(
                                [
                                    html.H5("Page 3: Graph analysis",
                                            className="card-title"),
                                    html.P("Graph analysis",
                                           className="card-text"),
                                    # Wrapping the button in a div for centering
                                    html.Div(
                                        dbc.Button(
                                            "Go to Page 3", color="primary", href="/page3"),
                                        style={"margin-top": "auto",
                                               "text-align": "center"}
                                    ),
                                ],
                                style={
                                    "display": "flex", "flex-direction": "column", "height": "100%"}
                            )
                        ],
                        style={"height": "100%"},
                    ),
                    width=4
                ),
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardBody(
                                [
                                    html.H5("Page 4: Upload and Process Dataset",
                                            className="card-title"),
                                    html.P("upload and process dataset",
                                           className="card-text"),
                                    # Wrapping the button in a div for centering
                                    html.Div(
                                        dbc.Button(
                                            "Go to Page 4", color="primary", href="/page4"),
                                        style={"margin-top": "auto",
                                               "text-align": "center"}
                                    ),
                                ],
                                style={
                                    "display": "flex", "flex-direction": "column", "height": "100%"}
                            )
                        ],
                        style={"height": "100%"},
                    ),
                    width=4
                ),
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardBody(
                                [
                                    html.H5("Page 5: comparaison with RNX",
                                            className="card-title"),
                                    html.P("comparaison with RNX",
                                           className="card-text"),
                                    # Wrapping the button in a div for centering
                                    html.Div(
                                        dbc.Button(
                                            "Go to Page 5", color="primary", href="/page5"),
                                        style={"margin-top": "auto",
                                               "text-align": "center"}
                                    ),
                                ],
                                style={
                                    "display": "flex", "flex-direction": "column", "height": "100%"}
                            )
                        ],
                        style={"height": "100%"},
                    ),
                    width=4
                ),
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardBody(
                                [
                                    html.H5("Page 6: Thesis paper",
                                            className="card-title"),
                                    html.P("Directional quality assessment indicators for nonlinear dimensionality reduction and data visualisation",
                                           className="card-text"),
                                    # Wrapping the button in a div for centering
                                    html.Div(
                                        dbc.Button(
                                            "Go to Page 6", color="primary", href="/page6"),
                                        style={"margin-top": "auto",
                                               "text-align": "center"}
                                    ),
                                ],
                                style={
                                    "display": "flex", "flex-direction": "column", "height": "100%"}
                            )
                        ],
                        style={"height": "100%"},
                    ),
                    width=4
                ),
            ],
            className="g-4",
            style={"margin": "0 5rem"}
        ),
        # add an image from the asset folder
        html.Img(src=app.get_asset_url('Screenshot 2024-12-14 at 17.54.03.png'),
                 style={"width": "100%", "margin-top": "2rem"}),
        # Footer
        html.Div(
            "Designed for the Thesis Dashboard by Pierre Remacle © 2024",
            style={"text-align": "center", "margin-top": "3rem",
                   "font-size": "0.9rem", "color": "gray"}
        ),
    ],
    style={"padding": "2rem"}
)

# Main Layout
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),  # URL for page navigation
    html.Div(id='page-content')  # Content rendered based on the current page
])

# Callback to switch between pages


@ app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname')]
)
def display_page(pathname):
    if pathname == '/':
        return front_page_layout
    elif pathname == '/page1':
        return page1_layout
    elif pathname == '/page2':
        return page2_layout
    elif pathname == '/page3':
        return page3_layout
    elif pathname == '/page4':
        return page4_layout
    elif pathname == '/page5':
        return page5_layout
    else:
        return html.Div(
            [
                html.H1("404", style={
                        "font-size": "6rem", "text-align": "center"}),
                html.P("Page Not Found", style={
                       "text-align": "center", "font-size": "1.5rem"}),
            ],
            style={"padding": "2rem"}
        )


# Register the callbacks for other pages
cache = DataCache()
register_callbacks_page1(app, cache)
register_callbacks_page2(app, cache)
register_callbacks_page3(app, cache)
register_callbacks_page4(app)
register_callbacks_page5(app, cache)

if __name__ == '__main__':
    app.run_server(debug=True)
