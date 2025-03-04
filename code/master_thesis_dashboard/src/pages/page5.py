import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import os
import numpy as np
import plotly.graph_objs as go

from sklearn.neighbors import NearestNeighbors
# Sample dataset options


def compute_rnx(X, X_reduced, steps):
    # Compute the original and reduced nearest neighbors
    rnx_graph = []
    neighbors_original = NearestNeighbors(n_neighbors=len(X_reduced)-1).fit(X)
    neighbors_reduced = NearestNeighbors(
        n_neighbors=len(X_reduced)-1).fit(X_reduced)

    original_indices = neighbors_original.kneighbors(return_distance=False)
    reduced_indices = neighbors_reduced.kneighbors(return_distance=False)
    print(original_indices)

    for n_neighbors in steps:

        print(n_neighbors)

        # Compute the RNX score by comparing neighborhood preservation
        intersection_counts = [
            len(set(original_indices[i][:n_neighbors]) & set(reduced_indices[i][:n_neighbors])) for i in range(X.shape[0])
        ]
        qnx = np.mean(intersection_counts) / n_neighbors
        rnx = (((X.shape[0] - 1) * qnx) - n_neighbors) / \
            (X.shape[0] - 1 - n_neighbors)
        rnx_graph.append(rnx)
    return rnx_graph


folders = [f for f in os.listdir("../") if f.endswith("_data")]
# sort foder based on second element divider by _ in folder name
folders.sort(key=lambda x: x.split("_")[1])

# Page layout
page5_layout = html.Div(
    [

        dbc.NavbarSimple(
            children=[
                # Navbar Items
                dbc.NavItem(dbc.NavLink("Home", href="/")),
                dbc.NavItem(dbc.NavLink("Page 1", href="/page1")),
                dbc.NavItem(dbc.NavLink("Page 2", href="/page2")),
                dbc.NavItem(dbc.NavLink("Page 3", href="/page3")),
                dbc.NavItem(dbc.NavLink("Page 4", href="/page4")),
                dbc.NavItem(dbc.NavLink("Page 5", href="/page5")),
            ],
        ),
        html.H3("Comparaison between RNX and the new method",
                style={"text-align": "center"}),

        # Multi-selection component
        dcc.Dropdown(
            id="multi-dataset-selector",
            options=[{"label": dataset, "value": dataset}
                     for dataset in folders],
            multi=True,
            placeholder="Select datasets...",
            style={"width": "80%", "margin": "0 auto"}
        ),
        # two graphs side by side
        html.Div([
            dcc.Graph(id="graph1", style={
                      "width": "45%", "display": "inline-block"}),
            dcc.Graph(id="graph2", style={
                      "width": "45%", "display": "inline-block"}),
            dcc.Graph(id="graph3", style={
                "width": "10%", "display": "inline-block"})
        ]),

        # Output section as a table
        html.Div(id="selected-datasets-output", style={"margin-top": "20px"}),
    ],
)


rnx_datas = []
new_rnx_datas = []
loaded_datasets = []


def register_callbacks_page5(app, cache):
    @app.callback(
        [
            Output("selected-datasets-output", "children"),
            Output("graph1", "figure"),
            Output("graph2", "figure"),
            Output("graph3", "figure"),
        ],
        [Input("multi-dataset-selector", "value")],
    )
    def update_page(selected_datasets):
        global rnx_datas, new_rnx_datas, loaded_datasets

        # If no datasets are selected, return default messages and empty figures
        if not selected_datasets:
            empty_figure = go.Figure()
            empty_figure.update_layout(showlegend=False)
            return (
                "No datasets selected.",
                empty_figure,
                empty_figure,
                empty_figure,
            )

        # Update datasets: Remove unselected datasets and load new original_indices
        to_remove = []

        for i, dataset in enumerate(loaded_datasets):
            if dataset not in selected_datasets:
                to_remove.append(i)

        for i in to_remove[::-1]:
            del rnx_datas[i]
            del new_rnx_datas[i]
            del loaded_datasets[i]

        for dataset in selected_datasets:
            if dataset not in loaded_datasets:
                cache.set_folder(dataset)
                data = cache.load_data()
                rnx_datas.append(data["rnx"])
                new_rnx_datas.append(data["means"])
                loaded_datasets.append(dataset)
        print(loaded_datasets)
        # Compute ranking and summary data for the table
        sum_rnx = [sum(rnx)/len(rnx) for rnx in rnx_datas]
        sum_new_rnx = [sum(new_rnx)/len(new_rnx) for new_rnx in new_rnx_datas]
        index_order_rnx = np.argsort(sum_rnx)[::-1]
        # reverse the order to have the best at the top
        index_order_new_rnx = np.argsort(sum_new_rnx)[::-1]

        # Create the table
        table = dbc.Table(
            [
                html.Thead(
                    html.Tr(
                        [
                            html.Th("Dataset"),
                            html.Th("RNX"),
                            html.Th("New RNX"),
                            html.Th("Rank RNX"),
                            html.Th("Rank New RNX"),
                        ]
                    )
                ),
                html.Tbody(
                    [
                        html.Tr(
                            [
                                html.Td(selected_datasets[i]),
                                html.Td(sum_rnx[i]),
                                html.Td(sum_new_rnx[i]),
                                html.Td(index_order_rnx[i] + 1),
                                html.Td(index_order_new_rnx[i] + 1),
                            ]
                        )
                        for i in range(len(selected_datasets))
                    ]
                ),
            ],
            bordered=True,
            striped=True,
            hover=True,
        )

        # Create the figures
        fig1 = go.Figure()
        fig2 = go.Figure()
        fig3 = go.Figure()

        for i, dataset in enumerate(selected_datasets):
            fig1.add_trace(
                go.Scatter(
                    x=np.arange(len(rnx_datas[i])),
                    y=rnx_datas[i],
                    mode="lines+markers",
                    name=dataset,
                )
            )
            fig2.add_trace(
                go.Scatter(
                    x=np.arange(2, len(new_rnx_datas[i])+2),
                    y=new_rnx_datas[i],
                    mode="lines+markers",
                    name=dataset,
                )
            )
            fig3.add_trace(
                go.Scatter(
                    x=[None], y=[None], mode="lines+markers", name=dataset
                )
            )

        for i in range(13):
            y_values = i / 10 - ((np.arange(100) - 1.7) / np.arange(100))
            fig2.add_trace(
                go.Scatter(
                    x=np.arange(100),
                    y=-y_values,
                    mode="lines",
                    line=dict(dash="dash", color="grey", width=1),
                    name=f"{1-(i/10)}",
                )
            )

        # Configure layouts for the figures
        fig1.update_layout(
            title="RNX Score",
            xaxis=dict(title="size of neighborhood", type="log"),
            yaxis=dict(title="Values"),
            template="plotly_white",
            showlegend=False,
        )
        max_of_new_rnx = max([max(new_rnx) for new_rnx in new_rnx_datas])
        fig2.update_layout(
            title="New Method Score",
            yaxis=dict(title="Values", range=[0, max_of_new_rnx+0.1]),
            xaxis=dict(title="length of path"),
            template="plotly_white",
            showlegend=False,
        )
        fig3.update_layout(
            showlegend=True,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            plot_bgcolor="white",
            margin=dict(l=0, r=0, t=0, b=0),
            legend=dict(
                orientation="h",
                x=0.5,
                xanchor="center",
                y=1.1,
                yanchor="top",
            ),
        )

        return table, fig1, fig2, fig3
