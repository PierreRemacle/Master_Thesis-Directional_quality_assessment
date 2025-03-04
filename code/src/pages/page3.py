

from data_cache import DataCache
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
import os
import dash_bootstrap_components as dbc
# Function Definitions (remains the same)


def path_rnx_distance_sorted(LD_data, HD_data, LD_paths, HD_paths):
    results = {}
    for index in range(len(LD_data)):
        results[index] = {}
        for i in range(len(LD_paths[index])):
            LD_path = [x for x in LD_paths[index][i] if x != -1][1:]
            HD_path = [x for x in HD_paths[index][i] if x != -1][1:]
            distance = levenshteinDistanceDP(LD_path, HD_path)
            path_len = len(LD_path)

            if path_len not in results[index]:
                results[index][path_len] = [distance]
            else:
                results[index][path_len].append(distance)

        results[index] = {k: np.mean(v)
                          for k, v in sorted(results[index].items())}

    with open("./..results_color.pkl", "wb") as fp:
        pickle.dump(results, fp)


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

    return -y, paths, quality_of_individual_links, frequence_of_individual_links


def rescale(LD_data, HD_data, LD_paths, HD_paths, results):
    longest_path_len = 0
    ys = np.zeros([len(LD_data), 500])

    for index, data in results.items():
        longest_path_len = max(longest_path_len, max(data.keys()))

    means = np.zeros(longest_path_len)
    colors = np.zeros(len(LD_data))

    for index in range(len(LD_data)):
        data = results[index]
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

        ys[index] = np.pad(-y, (0, 500 - len(y)), "constant")

    ys_transpose = ys[:index].T
    means = np.zeros(longest_path_len)

    for i in range(longest_path_len):
        tempo = np.trim_zeros(np.sort(ys_transpose[i]))
        means[i] = np.mean(tempo)

    for i in range(index+1):
        colors[i] = np.mean(ys[i][:len(means)] - means)

    scatter_colored_points = go.Scatter(
        x=LD_data[:, 0],
        y=LD_data[:, 1],
        mode="markers",
        marker=dict(
            size=5,
            color=colors,
            colorscale="Viridis",
            showscale=True,
        ),
        name="Colored Points",
    )

    rescaled_line_plot = go.Scatter(
        x=np.arange(2, len(means) + 2),
        y=means,
        mode="lines",
        line=dict(color="blue"),
        name="Rescaled Means",
    )

    return scatter_colored_points, rescaled_line_plot, LD_data, means


# Load data
# get all the folder that end with _data in the parent directory
folders = [f for f in os.listdir("../") if f.endswith("_data")]
current_folder = ""
with open("./current_folder.txt", "r") as f:
    current_folder = f.read()

# Create the layout for Dash app
page3_layout = html.Div([
    # Header with links to home , page 1 , page 2 , page 3 and page 4
    dbc.NavbarSimple(
        children=[
            # Folder Selector Dropdown on the Left
            dcc.Dropdown(
                id='folder-selector3',
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
        ],
        style={
            "display": "flex",  # Use Flexbox for layout
            "justify-content": "space-between",  # Align items to the left
            "padding": "0.5rem 1rem",  # Adjust padding for the navbar
        },
    ),
    dcc.Graph(id="dynamic-graph5", config={'staticPlot': False}),
    dcc.Graph(id="dynamic-graph6", config={'staticPlot': False}),
    # stores value of length of displayed path when clicking in graph3 , -1 if no click
    dcc.Store(id='n2', data=-1)

])

# Register callbacks for Page 2


def register_callbacks_page3(app, cache):
    """
    Register callbacks for updating the second graph with the selected path and its quality.
    """

    @ app.callback(
        # This will update the selected folder
        Output('folder-selector3', 'value'),
        Output('n2', 'data', allow_duplicate=True),
        # This listens for folder selection
        Input('folder-selector3', 'value'), prevent_initial_call=True
    )
    def update_folder(selected_folder):
        # Set the selected folder in the cache
        cache.set_folder(selected_folder)
        print(f"Selected folder: {selected_folder}")

        return selected_folder, -1

    @ app.callback(
        [Output('dynamic-graph6', 'figure'),
         Output('dynamic-graph5', 'figure')],
        [Input('dynamic-graph5', 'clickData'),
         Input('n2', 'data')]
    )
    def update_selected_path(clickData, n):
        """
        Update the graphs based on the selected path and its quality when a point is clicked.
        """
        # Initialize empty figures

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
        fig_path = go.Figure()
        fig_quality = go.Figure()
        # Default layout for the quality graph
        fig_quality.update_layout(
            title="Path Quality",
            xaxis=dict(title="Index"),
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

        # Handle case where no point is clicked
        if clickData is None:
            print("No click data")
            return fig_quality, fig_path

        print("Click data received")

        viridis = mpl.colormaps['coolwarm']
        # Extract the index of the clicked point
        sum_of_paths_quality = np.zeros(200)
        paths_quality_frequency = np.zeros(200)
        sum_of_quality = np.zeros([len(LD_data), len(LD_data)])
        sum_of_frequencies = np.zeros([len(LD_data), len(LD_data)])
        all_paths = []
        # for start_index in range(30):
        for start_index in range(len(LD_data)):
            print(start_index)

            fig_path = go.Figure()
            fig_quality = go.Figure()
            # Default layout for the quality graph
            fig_quality.update_layout(
                title="Path Quality",
                xaxis=dict(title="Index"),
                yaxis=dict(title="Quality", range=[0, 1]),
            )
            fig_path.add_trace(scatter)
            fig_quality.add_trace(rescaled)

            path_quality, paths, quality_of_individual_links, frequencies = paths_quality_from_one_start(
                start_index, LD_data, HD_data, LD_paths, HD_paths, graph, means, n, results
            )
            sum_of_paths_quality += np.pad(path_quality,
                                           (0, 200 - len(path_quality)), "constant")
            sum_of_quality += quality_of_individual_links

            paths_quality_frequency += np.pad(np.ones(
                len([x for x in path_quality if x != 0])), (0, 200 - len(path_quality)), "constant")
            sum_of_frequencies += frequencies
            all_paths.append(paths)
        quality_of_individual_links = sum_of_quality / sum_of_frequencies
        # absolute max value of the quality
        max_quality = np.nanmax(np.abs(quality_of_individual_links))
        print(max_quality)
    # Normalize the quality values
        if max_quality != 0:
            quality_of_individual_links = quality_of_individual_links / max_quality
    # Initialize a list to track added segments
        already_added_segments = []

    # Color map for path segments
    # rescale the quality to be between 0 and 1
        quality_of_individual_links = (quality_of_individual_links + 1) / 2

        # Add path segments to the path graph
        for paths in all_paths:
            for path in paths:

                # Filter out invalid points
                valid_path = [x for x in path if x != -1]
                # only show path that have the same length as n+2
                for j in range(len(valid_path)-1):
                    segment = (valid_path[j], valid_path[j+1])
                    # Avoid adding duplicate segments
                    if segment in already_added_segments:
                        continue
                    already_added_segments.append(segment)

                    # Extract coordinates for the segment
                    x_coords = [LD_data[segment[0], 0],
                                LD_data[segment[1], 0]]
                    y_coords = [LD_data[segment[0], 1],
                                LD_data[segment[1], 1]]

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
        path_quality = sum_of_paths_quality / paths_quality_frequency
        path_quality = [x for x in path_quality if x != 0]
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
        [Output('n2', 'data'),],
        [Input('dynamic-graph5', 'clickData'),
         Input('n2', 'data')]
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
