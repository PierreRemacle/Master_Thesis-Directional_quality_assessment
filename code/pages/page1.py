import os
import pickle
import numpy as np
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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


def paths_quality_from_one_start(start_index, LD_data, HD_data, LD_paths, HD_paths):
    results = np.zeros(len(LD_paths[start_index]))
    for i in range(len(LD_paths[start_index])):
        LD_path = [x for x in LD_paths[start_index][i] if x != -1][1:]
        HD_path = [x for x in HD_paths[start_index][i] if x != -1][1:]
        distance = levenshteinDistanceDP(LD_path, HD_path)
        results[i] = distance / len(LD_path)

    return results, LD_paths[start_index]


def rescale(LD_data, HD_data, LD_paths, HD_paths, results):
    global frames
    frames = []
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

        if index % 10 == 0 or index == len(LD_data)-1:
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

            frame = go.Frame(
                data=[scatter_colored_points, rescaled_line_plot],
                name=f"frame{index}",
            )
            frames.append(frame)


folders = [f for f in os.listdir("../") if f.endswith("_data")]

# Create the layout for Dash app
page1_layout = html.Div([

    dbc.NavbarSimple(
        children=[
            # Folder Selector Dropdown on the Left
            dcc.Dropdown(
                id='folder-selector1',
                options=[{'label': folder, 'value': folder}
                        for folder in folders],
                value=open("current_folder.txt", "r").read(),
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
    dcc.Graph(id="dynamic-graph"),
    dcc.Graph(id="dynamic-graph2"),
    dcc.Slider(
        id='frame-slider',
        min=0,
        max=300 - 1,
        step=1,
        # Show labels every 10th frame
        marks={i: f'{(i)*10}' for i in range(0, 300, 10)},
        value=0,
    ),
    html.Div([
        html.Button('Play', id='play-button', n_clicks=0),
        html.Button('Pause', id='pause-button', n_clicks=0),
    ], style={'margin-top': '20px'}),
    dcc.Interval(
        id='interval-component',
        interval=100,  # Update every 100ms (adjust for speed)
        n_intervals=0,
        disabled=True  # Initially disabled
    )
])
# Update Graph 1 based on slider
# global variable to store frames
frames = []


def register_callbacks_page1(app, cache):
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
    rescale(LD_data, HD_data, LD_paths, HD_paths, results)

    @app.callback(
        [Output('folder-selector1', 'value'),
         Output('frame-slider', 'max'),
         Output('frame-slider', 'marks')],
        [Input('folder-selector1', 'value')],
        prevent_initial_call=True
    )
    def update_folder(selected_folder):
        # Set the selected folder in the cache
        cache.set_folder(selected_folder)
        print(f"Selected folder: {selected_folder}")

        global data
        # Recompute frames based on new data, frame is a global variable
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
        print(data["folder"])
        rescale(LD_data, HD_data, LD_paths, HD_paths, results)

        # update the global variable

        # Update the frame slider max value
        max_frames = len(frames) - 1
        marks = {i: f'{(i)*10}' for i in range(0, max_frames, 10)}

        return selected_folder, max_frames, marks

    @app.callback(
        Output('dynamic-graph', 'figure'),
        [Input('frame-slider', 'value'),
         Input('folder-selector1', 'value')],
    )
    def update_graph_1(selected_frame, selected_folder):
        print("Updating graph 1")
        fig = go.Figure()
        # Load data based on the selected folder
        #
        frame = frames[selected_frame]

        fig = make_subplots(
            rows=1, cols=1,
            subplot_titles=("2D Scatter Plot",),
        )

        fig.add_trace(frame.data[0])

        fig.update_layout(
            title="2D Scatter Plot",
            xaxis=dict(title="LD X"),
            yaxis=dict(title="LD Y"),
        )

        return fig

    @app.callback(
        Output('dynamic-graph2', 'figure'),
        [Input('frame-slider', 'value'),
         Input('folder-selector1', 'value')],
    )
    def update_graph_2(selected_frame, selected_folder):
        # Load data based on the selected folder
        fig = go.Figure()
        frame = frames[selected_frame]
        fig = make_subplots(
            rows=1, cols=1,
            subplot_titles=("Rescaled Means Plot",),
        )

        fig.add_trace(frame.data[1])

        for i in range(0, 13):
            y_values = i / 10 - ((np.arange(100) - 1.7) /
                                 (np.arange(100)))
            fig.add_trace(
                go.Scatter(
                    x=np.arange(100),
                    y=-y_values,
                    mode="lines",
                    line=dict(dash="dash", color="grey", width=1),
                    name=f"Guide Line {i}",
                ),
                row=1, col=1
            )

        fig.update_layout(
            title="Rescaled Means Plot",
            xaxis=dict(title="Index"),
            yaxis=dict(title="Rescaled Means", range=[0, 1]),
        )

        return fig

    # Play/Pause callback to manage slider value
    @app.callback(
        Output('frame-slider', 'value'),
        [Input('play-button', 'n_clicks'), Input('pause-button', 'n_clicks')],
        [State('frame-slider', 'value')], prevent_initial_call=True
    )
    def play_pause(play_clicks, pause_clicks, current_value):
        # Enable/Disable the interval timer based on button clicks
        if play_clicks > pause_clicks:
            return (current_value + 1) % len(frames)
        return current_value

    # Callback for interval to update slider when play is active
    @app.callback(
        Output('interval-component', 'disabled'),
        [Input('play-button', 'n_clicks'), Input('pause-button', 'n_clicks')]
    )
    def toggle_interval(play_clicks, pause_clicks):
        if play_clicks > pause_clicks:
            return False  # Enable the interval
        return True  # Disable the interval when pause is clicked

    # Update the slider value based on the interval
    @app.callback(
        Output('frame-slider', 'value', allow_duplicate=True),
        [Input('interval-component', 'n_intervals')],
        [State('frame-slider', 'value')], prevent_initial_call=True
    )
    def update_slider(n_intervals, current_value):
        # Increment the slider value and loop back to the beginning when reaching the last frame
        if current_value + 1 >= len(frames):
            # stop looping when reaching the end
            return current_value
        return (current_value + 1) % len(frames)


if __name__ == '__main__':
    app.run_server(debug=True)
