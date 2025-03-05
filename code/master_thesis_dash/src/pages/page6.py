from dash import Dash, html
import dash_pdf
import requests
from pathlib import Path
import dash
import dash_bootstrap_components as dbc

dash._dash_renderer._set_react_version("18.2.0")

app = Dash(__name__)


# Alternatively, you can read a local PDF file
pdf_bytes = Path('../../../TFE_PIERRE.pdf').read_bytes()

page6_layout = html.Div([

    dbc.NavbarSimple(
        children=[
            # Navbar Items
            dbc.NavItem(dbc.NavLink("Home", href="/")),
            dbc.NavItem(dbc.NavLink("Page 1", href="/page1")),
            dbc.NavItem(dbc.NavLink("Page 2", href="/page2")),
            dbc.NavItem(dbc.NavLink("Page 3", href="/page3")),
            dbc.NavItem(dbc.NavLink("Page 4", href="/page4")),
            dbc.NavItem(dbc.NavLink("Page 5", href="/page5")),
            dbc.NavItem(dbc.NavLink("Page 6", href="/page6")),
        ],
    ),
    dash_pdf.PDF(
        id='pdf-viewer',
        # Pass the PDF content as bytes, you can also pass a URL
        data=pdf_bytes,
        # use these to customize the class names
        buttonClassName="",
        labelClassName="",
        controlsClassName="",
    )
])
