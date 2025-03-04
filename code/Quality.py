from Utils import *
knn = NearestNeighbors(n_neighbors=30)
nbrs = knn.fit(LD_data)
distances, indices = nbrs.kneighbors(LD_data)
graph = nbrs.kneighbors_graph(LD_data, mode="distance").toarray()
LD_path, HD_path = LDHDPath(knn, graph, 1, 2)
print(LD_path)
print(HD_path)

if len(HD_data.columns) == 4:
    fig = make_subplots(rows=1, cols=3, specs=[
                        [{'type': 'scene'}, {'type': 'scatter'}, {'type': 'scatter'}]])

    # Add the scatter plot for HD_data
    scatter_trace = go.Scatter3d(
        x=HD_data["0"], y=HD_data["1"], z=HD_data["2"],
        mode='markers', marker=dict(size=4, color=HD_data[HD_data.columns[0]], opacity=0.2),
        name='HD_data'
    )
    fig.add_trace(scatter_trace, row=1, col=1)

    # Add path traces for LD_path and HD_path
    for path, color in zip([LD_path, HD_path], ['red', 'green']):
        for i in range(len(path) - 1):
            path_trace = go.Scatter3d(
                x=[HD_data["0"][path[i]], HD_data["0"][path[i + 1]]],
                y=[HD_data["1"][path[i]], HD_data["1"][path[i + 1]]],
                z=[HD_data["2"][path[i]], HD_data["2"][path[i + 1]]],
                mode='lines', line=dict(color=color),
                name='Path'
            )
            fig.add_trace(path_trace, row=1, col=1)
    # Add callback function to handle point selection
     fig.add_trace(go.Scatter(
        x=LD_data["0"], y=LD_data["1"], mode='markers', name='LD_data'), row=1, col=2)
    for i in range(len(LD_path) - 1):
        fig.add_trace(go.Scatter(x=[LD_data["0"][LD_path[i]], LD_data["0"][LD_path[i + 1]]],
                                 y=[LD_data["1"][LD_path[i]],
                                     LD_data["1"][LD_path[i + 1]]],
                                 mode='lines', line=dict(color='red'),
                                 name='LD_path'), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=LD_data["0"], y=LD_data["1"], mode='markers', name='LD_data'), row=1, col=3)
    for i in range(len(HD_path) - 1):
        fig.add_trace(go.Scatter(x=[LD_data["0"][HD_path[i]], LD_data["0"][HD_path[i + 1]]],
                                 y=[LD_data["1"][HD_path[i]],
                                     LD_data["1"][HD_path[i + 1]]],
                                 mode='lines', line=dict(color='green'),
                                 name='HD_path'), row=1, col=3)
    fig.show()
else:
    fig = make_subplots(rows=1, cols=2, specs=[
                        [{'type': 'scatter'}, {'type': 'scatter'}]])
    fig.add_trace(go.Scatter(
        x=LD_data["0"], y=LD_data["1"], mode='markers', name='LD_data'), row=1, col=1)
    for i in range(len(LD_path) - 1):
        fig.add_trace(go.Scatter(x=[LD_data["0"][LD_path[i]], LD_data["0"][LD_path[i + 1]]],
                                 y=[LD_data["1"][LD_path[i]],
                                     LD_data["1"][LD_path[i + 1]]],
                                 mode='lines', line=dict(color='red'),
                                 name='LD_path'), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=LD_data["0"], y=LD_data["1"], mode='markers', name='LD_data'), row=1, col=2)
    for i in range(len(HD_path) - 1):
        fig.add_trace(go.Scatter(x=[LD_data["0"][HD_path[i]], LD_data["0"][HD_path[i + 1]]],
                                 y=[LD_data["1"][HD_path[i]],
                                     LD_data["1"][HD_path[i + 1]]],
                                 mode='lines', line=dict(color='green'),
                                 name='HD_path'), row=1, col=2)
    fig.show()


#
# print(LDHDCompare(LD_path, HD_path))

# plot the path
# plot two graphs
# if len(HD_data.columns) == 4:
#     panel = plt.figure()
#     ax = panel.add_subplot(131, projection='3d')
#     print(HD_data["0"])
#     print(HD_data["1"])
#     print(HD_data["2"])
#     ax.scatter(HD_data["0"], HD_data["1"],
#                HD_data["2"], c=HD_data[HD_data.columns[0]],  marker='o', s=4)
#     for i in range(len(LD_path) - 1):
#         plt.plot([HD_data["0"][LD_path[i]], HD_data["0"][LD_path[i + 1]]],
#                  [HD_data["1"][LD_path[i]], HD_data["1"][LD_path[i + 1]]],
#                  [HD_data["2"][LD_path[i]], HD_data["2"][LD_path[i + 1]]], 'r')
#     for i in range(len(HD_path) - 1):
#         plt.plot([HD_data["0"][HD_path[i]], HD_data["0"][HD_path[i + 1]]],
#                  [HD_data["1"][HD_path[i]], HD_data["1"][HD_path[i + 1]]],
#                  [HD_data["2"][HD_path[i]], HD_data["2"][HD_path[i + 1]]], 'g')
#     panel.add_subplot(132)
#     plt.scatter(LD_data["0"], LD_data["1"])
#     for i in range(len(LD_path) - 1):
#         plt.plot([LD_data["0"][LD_path[i]], LD_data["0"][LD_path[i + 1]]],
#                  [LD_data["1"][LD_path[i]], LD_data["1"][LD_path[i + 1]]], 'r')
#     panel.add_subplot(133)
#     plt.scatter(LD_data["0"], LD_data["1"])
#     for i in range(len(HD_path) - 1):
#         plt.plot([LD_data["0"][HD_path[i]], LD_data["0"][HD_path[i + 1]]],
#                  [LD_data["1"][HD_path[i]], LD_data["1"][HD_path[i + 1]]], 'g')
#     plt.show()
# else:
#     panel = plt.figure()
#     panel.add_subplot(121)
#
#     plt.scatter(LD_data["0"], LD_data["1"])
#     for i in range(len(LD_path) - 1):
#         plt.plot([LD_data["0"][LD_path[i]], LD_data["0"][LD_path[i + 1]]],
#                  [LD_data["1"][LD_path[i]], LD_data["1"][LD_path[i + 1]]], 'r')
#     panel.add_subplot(122)
#     plt.scatter(LD_data["0"], LD_data["1"])
#     for i in range(len(HD_path) - 1):
#         plt.plot([LD_data["0"][HD_path[i]], LD_data["0"][HD_path[i + 1]]],
#                  [LD_data["1"][HD_path[i]], LD_data["1"][HD_path[i + 1]]], 'g')
#     plt.show()
