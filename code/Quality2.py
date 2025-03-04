
import networkx as nx
import matplotlib.pyplot as plt
import dash
from dash import dcc, html, callback_context
from dash.dependencies import Input, Output
from Utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dash.exceptions import PreventUpdate
# Compute paths
# knn = NearestNeighbors(n_neighbors=30, p=2, algorithm="auto")
# nbrs = knn.fit(LD_data)
# distances, indices = nbrs.kneighbors(LD_data)
# graph = nbrs.kneighbors_graph(LD_data, mode="distance").toarray()
graph = delaunay_graph(LD_data)
# display graph


def show_graph_with_labels(adjacency_matrix, mylabels):
    rows, cols = np.where(adjacency_matrix != 0)
    edges = zip(rows.tolist(), cols.tolist())
    gr = nx.Graph()
    gr.add_edges_from(edges)
    nx.draw(gr, node_size=500, labels=mylabels, with_labels=True)
    plt.show()


# show_graph_with_labels(graph, {i: str(i) for i in range(len(graph))})

LD_paths, HD_paths = LDHDPathAll(graph, 1)
LD_path = LD_paths[2]
HD_path = HD_paths[2]
LD_path = LD_path[LD_path != -1]
HD_path = HD_path[HD_path != -1]
# Create Dash app
app = dash.Dash(__name__)

# Define app layout
app.layout = html.Div([
    html.H1("Path Visualization"),
    dcc.Input(id='input1', value=1, type='number'),
    dcc.Input(id='input2', value=3, type='number'),

    html.Div(children=[
        dcc.Graph(id='HD-graph-LD-path',
                  style={'display': 'inline-block', 'width': '50%'}),
        dcc.Graph(id='HD-graph-HD-path',
                  style={'display': 'inline-block', 'width': '50%'}),],
             style={'display': 'flex', 'height': '75vh'}),
    html.Div(children=[
        dcc.Markdown("0", id='cost-LD-path',
                     style={'display': 'inline-block', 'width': '50%'}),
        dcc.Markdown("0", id='cost-HD-path',
                     style={'display': 'inline-block', 'width': '50%'})],
             style={'display': 'flex'}),
    html.Div(children=[
        dcc.Graph(id='LD-graph-LD-path',
                  style={'display': 'inline-block', 'width': '50%'}),
        dcc.Graph(id='LD-graph-HD-path',
                  style={'display': 'inline-block', 'width': '50%'}),],
             style={'display': 'flex', 'height': '75vh'}),
    html.Div(children=[
        dcc.Markdown("0", id='LD-path',
                     style={'display': 'inline-block', 'width': '50%'}),
        dcc.Markdown("0", id='HD-path',
                     style={'display': 'inline-block', 'width': '50%'})],
             style={'display': 'flex'}),
])


def plot_HD_graph_LD_path(start, end):
    start = int(start)
    end = int(end)
    LD_paths, HD_paths = LDHDPathAll(graph, start)
    LD_path = LD_paths[end]
    HD_path = HD_paths[end]
    LD_path = LD_path[LD_path != -1]
    HD_path = HD_path[HD_path != -1]
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Scatter3d(
        x=HD_data[:, 0], y=HD_data[:, 1], z=HD_data[:, 2],
        mode='markers', marker=dict(size=4, opacity=0.2),
        name='HD_data'))
    for path, color in zip([LD_path], ['red']):
        for i in range(len(path) - 1):
            path_trace = go.Scatter3d(
                x=[HD_data[:, 0][path[i]], HD_data[:, 0][path[i + 1]]],
                y=[HD_data[:, 1][path[i]], HD_data[:, 1][path[i + 1]]],
                z=[HD_data[:, 2][path[i]], HD_data[:, 2][path[i + 1]]],
                mode='lines', line=dict(color=color),
                name='Path'
            )
            fig.add_trace(path_trace)
    fig.update_layout(showlegend=False)
    return fig


def plot_HD_graph_HD_path(start, end):
    end = int(end)
    start = int(start)
    LD_paths, HD_paths = LDHDPathAll(graph, start)
    LD_path = LD_paths[end]
    HD_path = HD_paths[end]
    LD_path = LD_path[LD_path != -1]
    HD_path = HD_path[HD_path != -1]
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Scatter3d(
        x=HD_data[:, 0], y=HD_data[:, 1], z=HD_data[:, 2],
        mode='markers', marker=dict(size=4, opacity=0.2),
        name='HD_data'
    ))
    for path, color in zip([HD_path], ['green']):
        for i in range(len(path) - 1):
            path_trace = go.Scatter3d(
                x=[HD_data[:, 0][path[i]], HD_data[:, 0][path[i + 1]]],
                y=[HD_data[:, 1][path[i]], HD_data[:, 1][path[i + 1]]],
                z=[HD_data[:, 2][path[i]], HD_data[:, 2][path[i + 1]]],
                mode='lines', line=dict(color=color),
                name='Path'
            )
            fig.add_trace(path_trace)
    fig.update_layout(showlegend=False)
    return fig

# Define callback to update input1 and input2 when a point is clicked on the graph


@ app.callback(
    [Output('cost-LD-path', 'children'),
     Output('cost-HD-path', 'children'),
     Output('LD-path', 'children'),
     Output('HD-path', 'children')],
    [Input('input1', 'value'),
     Input('input2', 'value')]
)
def update_cost(indice1, indice2):
    indice1 = int(indice1)
    indice2 = int(indice2)
    LD_paths, HD_paths = LDHDPathAll(graph, indice1)
    LD_path = LD_paths[indice2]
    HD_path = HD_paths[indice2]
    LD_path = LD_path[LD_path != -1]
    HD_path = HD_path[HD_path != -1]
    return 0, 0, 0, 0


@ app.callback(
    [Output('input1', 'value'),
     Output('input2', 'value')],
    [Input('HD-graph-LD-path', 'clickData'),
     Input('HD-graph-HD-path', 'clickData'),
     Input('LD-graph-LD-path', 'clickData'),
     Input('LD-graph-HD-path', 'clickData'),
     Input('input1', 'value'),],
    prevent_initial_call=True
)
def update_input(clickData1, clickData2, clickData3, clickData4, input1):
    ctx = callback_context
    index = 0
    if not ctx.triggered:
        raise PreventUpdate
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        point = None
        if button_id == 'HD-graph-LD-path':
            point = clickData1['points'][0]
            index = HD_data[(HD_data[:, 0] == point["x"])
                            & (HD_data[:, 1] == point["y"]) & (HD_data[:, 2] == point["z"])].index[0]
        elif button_id == 'HD-graph-HD-path':
            point = clickData2['points'][0]
            index = HD_data[(HD_data[:, 0] == point["x"])
                            & (HD_data[:, 1] == point["y"]) & (HD_data[:, 2] == point["z"])].index[0]
        elif button_id == 'LD-graph-LD-path':
            point = clickData3['points'][0]
            index = LD_data[(LD_data[:, 0] == point["x"]) & (
                LD_data[:, 1] == point["y"])][0]
        elif button_id == 'LD-graph-HD-path':
            point = clickData4['points'][0]
            point = [point["x"], point["y"]]
            index = -1
            for i in range(len(LD_data)):
                if np.array_equal(LD_data[i], point):
                    index = i
        # find point index in HD_data

        return index, input1

# Define callback function to update camera position of one subplot when the other subplot is rotated


@ app.callback(
    Output('HD-graph-LD-path', 'figure', allow_duplicate=True),
    [Input('HD-graph-HD-path', 'relayoutData'),
     Input('HD-graph-LD-path', 'figure'),
     Input('input1', 'value'),
     Input('input2', 'value')],
    prevent_initial_call=True
)
def update_camera(HD_relayout, LD_relayout, indice1, indice2):
    if HD_relayout and 'scene.camera' in HD_relayout:
        fig = plot_HD_graph_LD_path(indice1, indice2)
        fig.update_layout(scene_camera=HD_relayout['scene.camera'])
        return fig
    else:
        return LD_relayout


@ app.callback(
    Output('HD-graph-HD-path', 'figure', allow_duplicate=True),
    [Input('HD-graph-LD-path', 'relayoutData'),
     Input('HD-graph-HD-path', 'figure'),
     Input('input1', 'value'),
     Input('input2', 'value')],
    prevent_initial_call=True
)
def update_camera(LD_relayout, HD_relayout, indice1, indice2):
    if LD_relayout and 'scene.camera' in LD_relayout:
        fig = plot_HD_graph_HD_path(indice1, indice2)
        fig.update_layout(scene_camera=LD_relayout['scene.camera'])
        return fig
    else:
        return HD_relayout


@ app.callback(
    Output('HD-graph-LD-path', 'figure'),
    [Input('input1', 'value'),
     Input('input2', 'value')])
def update_graph(indice1, indice2):
    indice1 = int(indice1)
    indice2 = int(indice2)
    LD_paths, HD_paths = LDHDPathAll(graph, indice1)
    LD_path = LD_paths[indice2]
    HD_path = HD_paths[indice2]
    LD_path = LD_path[LD_path != -1]
    HD_path = HD_path[HD_path != -1]
    return plot_HD_graph_LD_path(indice1, indice2)


@ app.callback(
    Output('HD-graph-HD-path', 'figure'),
    [Input('input1', 'value'),
     Input('input2', 'value')])
def update_graph(indice1, indice2):
    indice1 = int(indice1)
    indice2 = int(indice2)
    LD_paths, HD_paths = LDHDPathAll(graph, indice1)
    LD_path = LD_paths[indice2]
    HD_path = HD_paths[indice2]
    LD_path = LD_path[LD_path != -1]
    HD_path = HD_path[HD_path != -1]
    return plot_HD_graph_HD_path(indice1, indice2)


@ app.callback(
    Output('LD-graph-LD-path', 'figure'),
    [Input('input1', 'value'),
     Input('input2', 'value')])
def update_graph(indice1, indice2):
    indice1 = int(indice1)
    indice2 = int(indice2)
    LD_paths, HD_paths = LDHDPathAll(graph, indice1)
    LD_path = LD_paths[indice2]
    HD_path = HD_paths[indice2]
    LD_path = LD_path[LD_path != -1]
    HD_path = HD_path[HD_path != -1]
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Scatter(
        x=LD_data[:, 0], y=LD_data[:, 1], mode='markers', name='LD_data'))
    for i in range(len(LD_path) - 1):
        fig.add_trace(go.Scatter(x=[LD_data[LD_path[i]][0], LD_data[LD_path[i + 1]][0]],
                                 y=[LD_data[LD_path[i]][1],
                                    LD_data[LD_path[i + 1]][1]],
                                 mode='lines', line=dict(color='red'),
                                 name='LD_path'))
    fig.update_layout(showlegend=False)
    return fig


@ app.callback(
    Output('LD-graph-HD-path', 'figure'),
    [Input('input1', 'value'),
     Input('input2', 'value')])
def update_graph(indice1, indice2):
    indice1 = int(indice1)
    indice2 = int(indice2)
    LD_paths, HD_paths = LDHDPathAll(graph, indice1)
    LD_path = LD_paths[indice2]
    HD_path = HD_paths[indice2]
    LD_path = LD_path[LD_path != -1]
    HD_path = HD_path[HD_path != -1]
    print(LD_path)
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Scatter(
        x=LD_data[:, 0], y=LD_data[:, 1], mode='markers', name='LD_data'))
    for i in range(len(HD_path) - 1):
        fig.add_trace(go.Scatter(x=[LD_data[:, 0][LD_path[i]], LD_data[:, 0][LD_path[i + 1]]],
                                 y=[LD_data[:, 1][LD_path[i]],
                                     LD_data[:, 1][LD_path[i + 1]]],
                                 mode='lines', line=dict(color='red'), name='LD_path'))
    fig.update_layout(showlegend=False)
    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
