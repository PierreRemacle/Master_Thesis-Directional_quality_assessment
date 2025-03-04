from Utils import *

from dash import Dash, html, dcc, callback, Output, Input
import plotly.express as px
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import alive_progress as ap
app = Dash(__name__)

app.layout = html.Div([
    html.H1(children='Title of Dash App', style={'textAlign': 'center'}),
    dcc.Graph(id='graph-content')
])


@callback(
    Output('graph-content', 'figure'),
    Input('graph-content', 'clickData')
)
def update_graph(clickData):
    plt = make_subplots(rows=1, cols=2)
    plt.add_trace(go.Scatter(
        x=LD_data[:, 0], y=LD_data[:, 1], mode='markers', name='LD_data'), row=1, col=1)
    plt.add_trace(go.Scatter(
        x=LD_data[:, 0], y=LD_data[:, 1], mode='markers', name='HD_data'), row=1, col=2)
    if clickData is not None:
        start = clickData['points'][0]['pointIndex']

        LD_path, localisation_of_errors = all_path(
            LD_data, start)
        for i in range(len(localisation_of_errors)):
            path = LD_path[i][LD_path[i] != -1]
        for i in range(len(localisation_of_errors)):
            for j in range(len(localisation_of_errors[i])):
                if localisation_of_errors[i][j] > 0:
                    plt.add_trace(go.Scatter(x=[LD_data[i][0], LD_data[j][0]], y=[LD_data[i][1], LD_data[j][1]],
                                             mode='lines', name='lines+markers', line=dict(color="blue", width=localisation_of_errors[i][j] / localisation_of_errors.max() * 3)), row=1, col=2)
        colors = ['red', 'green', 'blue', 'yellow', 'purple',
                  'orange', 'pink', 'brown', 'black', 'grey']
        print("ah")
        already_seen = []
        for a, path in enumerate(LD_path):
            path = path[path != -1]
            for i in range(len(path) - 1):
                if (path[i], path[i+1]) not in already_seen:
                    plt.add_trace(go.Scatter(x=[LD_data[path[i]][0], LD_data[path[i+1]][0]], y=[LD_data[path[i]][1], LD_data[path[i+1]][1]], mode='lines',
                                             marker=dict(color=colors[1]), name='start'), row=1, col=1)
                    already_seen.append((path[i], path[i+1]))
    return plt


def all_path(LD_data, start):
    localisation_of_errors = np.zeros((len(LD_data), len(LD_data)))
    frequence_of_path = np.ones((len(LD_data), len(LD_data)))

    graph = delaunay_graph(LD_data)

    LD_paths, HD_paths = LDHDPathAll(graph, start)
    for end in range(len(LD_data)):
        if start != end:
            LD_path = LD_paths[end]
            HD_path = HD_paths[end]
            LD_path = LD_path[LD_path != -1]
            HD_path = HD_path[HD_path != -1]
            for k in range(len(LD_path)-1):
                first = min(LD_path[k], LD_path[k+1])
                second = max(LD_path[k], LD_path[k+1])
                frequence_of_path[first][second] += 1
                if LD_path[k+1] != HD_path[k+1]:
                    localisation_of_errors[first][second] += 1
    return LD_paths, localisation_of_errors/frequence_of_path


if __name__ == '__main__':
    app.run(debug=True)
