
import plotly.express as px
from re import L
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from dash.dependencies import Input, Output, State
from dash import dcc, html
import dash
import numpy as np
import pickle
import sys
import os
from Utils import *
import matplotlib as mpl
import dash_bootstrap_components as dbc
# Function Definitions (remains the same)


def paths_quality_from_one_start(start_index, LD_data, HD_data, LD_paths, HD_paths, graph, real_y, n, results):
    quality_of_individual_links = np.zeros([len(LD_data), len(LD_data)])
    frequence_of_individual_links = np.zeros([len(LD_data), len(LD_data)])
    paths = LD_paths[start_index]
    paths = np.array(paths, dtype=int)
    longest_path_len = 0
    ys = np.zeros([len(LD_data), 500])

    for i in range(len(paths)):
        path = paths[i]
        path = [x for x in path if x != -1]
        HD_path = HD_paths[start_index][i]
        HD_path = [x for x in HD_path if x != -1]
        if len(path) == n or n == -1:
            for j in range(len(path)-1):
                q = levenshteinDistanceDP(HD_path, path)
                # apply rnx transformation,
                q = -(q / len(path) - ((len(path) - 1.7) / len(path)))
                # this_y = ((-real_y[len(path) - 2] *
                #            len(path)) + (len(path) - 1.7))
                q_diff = q - real_y[len(path) - 2]
                # q_diff = (q-this_y)
                for k in range(j+1):
                    quality_of_individual_links[path[k],
                                                path[k + 1]] += q_diff
                    frequence_of_individual_links[path[k],
                                                  path[k + 1]] += 1

    quality_of_individual_links = quality_of_individual_links / \
        frequence_of_individual_links
    for index, data in results.items():
        longest_path_len = max(longest_path_len, max(data.keys()))

    means = np.zeros(longest_path_len)
    colors = np.zeros(len(LD_data))

    data = results[start_index]
    data = {k: v for k, v in sorted(
        data.items(), key=lambda item: item[0])}
    keys = list(data.keys())
    for i in range(len(keys)-1, -1, -1):
        data[keys[i]+2] = data[keys[i]]
    del data[1]
    del data[0]
    data = {k: v for k, v in sorted(
        data.items(), key=lambda item: item[0])}
    y = np.array(list(data.values())) / list(data.keys()) - \
        ((np.array(list(data.keys())) - 1.7) / (np.array(list(data.keys()))))

    return -y, paths, quality_of_individual_links


# get all the folder that end with _data in the parent directory
folders = [f for f in os.listdir("../") if f.endswith("_data")]

current_folder = ""
with open("./current_folder.txt", "r") as f:
    current_folder = f.read()
# Create the layout for Dash app
page2_layout = html.Div([

    dbc.NavbarSimple(
        children=[
            # Folder Selector Dropdown on the Left
            dcc.Dropdown(
                id='folder-selector2',
                options=[{'label': folder, 'value': folder}
                        for folder in folders],
                value=current_folder,
                style={
                    "width": "250px",  # Adjust the width of the dropdown
                    "margin-left": "-8rem",  # Add some margin to the left
                    "flex-shrink": 0,  # Prevent shrinking when space is tight
                    # put the dropdown on the left side of the navbar
                    "margin-right": "auto",


                },
            ),

            # Navbar Items
            dbc.NavItem(dbc.NavLink("Home", href="/")),
            dbc.NavItem(dbc.NavLink("Page 1", href="/page1")),
            dbc.NavItem(dbc.NavLink("Page 2", href="/page2")),
            dbc.NavItem(dbc.NavLink("Page 3", href="/page3")),
            dbc.NavItem(dbc.NavLink("Page 4", href="/page4")),
            dbc.NavItem(dbc.NavLink("Page 5", href="/page5")),
            dbc.NavItem(dbc.NavLink("Page 6", href="/page6")),
        ],
        style={
            "display": "flex",  # Use Flexbox for layout
            "justify-content": "space-between",  # Align items to the left
            "padding": "0.5rem 1rem",  # Adjust padding for the navbar
        },
    ),
    html.Div(
        [
            dcc.Graph(
                id="dynamic-graph4",
                config={'staticPlot': False},
                style={"height": "80vh", "width": "80vh"}
            ),
            dcc.Graph(
                id="dynamic-graph3",
                config={'staticPlot': False},
                style={"height": "80vh", "width": "80vh"}
            ),
        ],
        style={
            "display": "flex",
            "flex-wrap": "wrap",
            "justify-content": "center",
            "gap": "1rem",
        }
    ),
    # stores value of length of displayed path when clicking in graph3 , -1 if no click
    dcc.Store(id='n', data=-1)

])

# Register callbacks for Page 2


def register_callbacks_page2(app, cache):
    """
    Register callbacks for updating the second graph with the selected path and its quality.
    """

    @app.callback(
        # This will update the selected folder
        Output('folder-selector2', 'value'),
        Output('n', 'data', allow_duplicate=True),
        # This listens for folder selection
        Input('folder-selector2', 'value'), prevent_initial_call=True
    )
    def update_folder(selected_folder):
        # Set the selected folder in the cache
        cache.set_folder(selected_folder)
        print(f"Selected folder: {selected_folder}")

        return selected_folder, -1

    @ app.callback(
        [Output('dynamic-graph3', 'figure'),
         Output('dynamic-graph4', 'figure')],
        [Input('dynamic-graph4', 'clickData'),
         Input('n', 'data'),
         Input('folder-selector2', 'value')]
    )
    def update_selected_path(clickData, n, selected_folder):
        """
        Update the graphs based on the selected path and its quality when a point is clicked.
        """

        data = cache.load_data()
        LD_data = data["LD_data"]
        HD_data = data["HD_data"]
        LD_paths = data["LD_paths"]
        HD_paths = data["HD_paths"]
        graph = data["graph"]
        results = data["results"]
        scatter = data["scatter"]
        rescaled = data["rescaled"]
        means = data["means"]

        # Initialize empty figures
        fig_path = go.Figure()
        fig_quality = go.Figure()

        # Default layout for the quality graph
        fig_quality.update_layout(
            title="Path Quality",
            xaxis=dict(title="Path Length", range=[2, rescaled.x.max()]),
            yaxis=dict(title="Quality", range=[0, 1]),
        )
        fig_path.add_trace(scatter)
        fig_quality.add_trace(rescaled)

        # Add guideline traces to the quality graph

        for i in range(13):
            y_values = i / 10 - ((np.arange(100) - 1.7) / np.arange(100))
            fig_quality.add_trace(
                go.Scatter(
                    x=np.arange(100),
                    y=-y_values,
                    mode="lines",
                    line=dict(dash="dash", color="grey", width=1),
                    name=f"Guide Line {i}",
                )
            )
        # add a dot on the n point
        if n != -1:
            fig_quality.add_trace(
                go.Scatter(
                    x=[n],
                    y=[means[n-2]],
                    mode="markers",
                    marker=dict(
                        size=10,
                        color="red",
                    ),
                    name="Selected Point",
                )
            )

        # Handle case where no point is clicked
        if clickData is None:
            print("No click data")
            return fig_quality, fig_path

        print("Click data received")

        # Extract the index of the clicked point
        clicked_point = clickData['points'][0]
        start_index = clicked_point['pointIndex']

        # Compute path quality and related data
        path_quality, paths, quality_of_individual_links = paths_quality_from_one_start(
            start_index, LD_data, HD_data, LD_paths, HD_paths, graph, means, n, results)

        # absolute max value of the quality
        max_quality = np.nanmax(np.abs(quality_of_individual_links))
        # Normalize the quality values
        print("max = ", max_quality)
        if max_quality != 0:
            quality_of_individual_links = quality_of_individual_links / max_quality
        # Initialize a list to track added segments
        already_added_segments = []

        # Color map for path segments
        viridis = mpl.colormaps['coolwarm']
        # rescale the quality to be between 0 and 1
        quality_of_individual_links = (quality_of_individual_links + 1) / 2

        # Add path segments to the path graph
        for path in paths:
            # Filter out invalid points
            valid_path = [x for x in path if x != -1]
            # only show path that have the same length as n+2
            if len(valid_path) == n or n == -1:

                for j in range(len(valid_path)-1):
                    segment = (valid_path[j], valid_path[j+1])
                    # Avoid adding duplicate segments
                    if segment in already_added_segments:
                        continue
                    already_added_segments.append(segment)

                    # Extract coordinates for the segment
                    x_coords = [LD_data[segment[0], 0], LD_data[segment[1], 0]]
                    y_coords = [LD_data[segment[0], 1], LD_data[segment[1], 1]]

                    segment_color = quality_of_individual_links[segment[0], segment[1]]

                    # Convert color map value to RGB
                    color = viridis(segment_color)
                    rgb = f"rgb({int(color[0] * 255)},"
                    rgb += f"{int(color[1] * 255)},"
                    rgb += f"{int(color[2] * 255)})"

                    # Add the segment to the path graph
                    fig_path.add_trace(
                        go.Scatter(
                            x=x_coords,
                            y=y_coords,
                            mode="lines",
                            line=dict(
                                color=rgb,
                                width=1
                            ),
                            name="Path",
                            showlegend=False,
                            text=f"{segment_color:.2f}",
                        )
                    )

        # Add the overall path quality to the quality graph
        fig_quality.add_trace(
            go.Scatter(
                x=np.arange(len(path_quality)) + 2,
                y=path_quality,
                mode="lines",
                line=dict(color='red'),
                name="Path Quality",
            )
        )

        # Update layout for the path graph
        fig_path.update_layout(
            title="Path and Quality from Selected Point",
            xaxis=dict(title="X Coordinate"),
            yaxis=dict(title="Y Coordinate"),
        )

        return fig_quality, fig_path

    @ app.callback(
        [Output('n', 'data'),],
        [Input('dynamic-graph3', 'clickData'),
         Input('n', 'data')]
    )
    def update_n(clickData, n):
        """
        Update the stored value of n when a point is clicked.
        """
        if clickData is None:
            return [-1]

        print("Click data received")
        print(clickData)
        clicked_point = clickData['points'][0]
        start_index = clicked_point['x']
        print(start_index)
        if n == start_index:
            return [-1]
        return [start_index]


# Run the Dash app
if __name__ == '__main__':
    app = dash.Dash(__name__)
    app.layout = page2_layout
    register_callbacks_page2(app)
    app.run_server(debug=True)
