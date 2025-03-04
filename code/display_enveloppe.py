from Utils import *
import sys
import numpy
import matplotlib.pyplot as plt
from alive_progress import alive_bar

from scipy.spatial import Delaunay

folder = 't-SNE_MNIST_data'
# load data
LD_data = np.load('./'+folder+'/LD_data.npy')
HD_data = np.load('./'+folder+'/HD_data.npy')
print(len(LD_data))
# load paths
HD_paths = np.load('./'+folder+'/HD_all_paths.npy')
HD_paths_2 = np.load('./'+folder+'/HD_all_paths_2.npy')
LD_paths = np.load('./'+folder+'/LD_all_paths.npy')
LD_paths_2 = np.load('./'+folder+'/LD_all_paths_2.npy')
# load distance matrix
LD_distance_matrix = np.load('./'+folder+'/LD_distance_matrix.npy')
HD_distance_matrix = np.load('./'+folder+'/HD_distance_matrix.npy')

enveloppe, all_index_enveloppe = multiple_enveloppe(LD_data, 3)
print(enveloppe)
total = 0
for env in enveloppe:
    print(len(env))
    total += len(env)
print(total)

# close the loop
# plt = make_subplots(rows=2, cols=2)
fig, axs = plt.subplots(1, 1)

axs.plot(LD_data[:, 0], LD_data[:, 1], 'o')
# plt.add_trace(go.Scatter(x=LD_data[:, 0], y=LD_data[:, 1], mode='markers'))
for index_enveloppe in all_index_enveloppe:
    index_enveloppe.append(index_enveloppe[0])
    index_enveloppe = np.array(index_enveloppe)
    for i in range(len(index_enveloppe)-1):
        # plt.add_trace(go.Scatter(x=(LD_data[index_enveloppe[i]][0], LD_data[index_enveloppe[i+1]][0]), y=(
        #     LD_data[index_enveloppe[i]][1], LD_data[index_enveloppe[i+1]][1]), mode='lines'))
        axs.plot([LD_data[index_enveloppe[i]][0], LD_data[index_enveloppe[i+1]][0]], [
            LD_data[index_enveloppe[i]][1], LD_data[index_enveloppe[i+1]][1]], 'r')


xss, yss, indexs = enveloppe_of_cluster(LD_data)
for index in indexs:
    print(LD_data[index][:, 0])
    # plt.add_trace(go.Scatter(
    #     x=LD_data[index][:, 0], y=LD_data[index][:, 1], mode='markers'), row=2, col=1)

# rescale LD_data to (0, 100),(0, 100)
LD_data = np.array(LD_data)
min = np.min(LD_data, axis=0)
max = np.max(LD_data, axis=0)
LD_data_rescale = LD_data - np.min(LD_data, axis=0)
LD_data_rescale = LD_data_rescale / np.max(LD_data_rescale, axis=0) * 100
alpha = 1
concave_hull, edge_points, _ = alpha_shape(LD_data_rescale, alpha=alpha)
plt.show()
fig, axs = plt.subplots(1, 1)
# plot concave hull

# plt.add_trace(go.Scatter(
#     x=LD_data[:, 0], y=LD_data[:, 1], mode='markers'), row=2, col=2)
axs.plot(LD_data[:, 0], LD_data[:, 1], 'o')
total = 0
if concave_hull.geom_type == 'Polygon':
    x, y = concave_hull.exterior.xy
    print(len(x))
    total += len(x)
    x = np.array(x)
    y = np.array(y)
    x = x / 100 * (max[0] - min[0]) + min[0]
    y = y / 100 * (max[1] - min[1]) + min[1]
    # plt.add_trace(go.Scatter(x=x, y=y, mode='lines'), row=2, col=2)
    axs.plot(x, y, 'r')
else:

    for polygon in concave_hull.geoms:
        x, y = polygon.exterior.xy
        print(len(x))
        total += len(x)
        x = np.array(x)
        y = np.array(y)
        x = x / 100 * (max[0] - min[0]) + min[0]
        y = y / 100 * (max[1] - min[1]) + min[1]
        # plt.add_trace(go.Scatter(x=x, y=y, mode='lines'), row=2, col=2)
        axs.plot(x, y, 'r')
print(total)
plt.show()


# # create grah for edge_points
#
# graph = {}
# for i in range(len(edge_points)):
#     if (edge_points[i][0][0], edge_points[i][0][1]) not in graph.keys():
#         graph[(edge_points[i][0][0], edge_points[i][0][1])] = set()
#     if (edge_points[i][1][0], edge_points[i][1][1]) not in graph.keys():
#         graph[(edge_points[i][1][0], edge_points[i][1][1])] = set()
#     graph[(edge_points[i][0][0], edge_points[i][0][1])].add(
#         (edge_points[i][1][0], edge_points[i][1][1]))
#     graph[(edge_points[i][1][0], edge_points[i][1][1])].add(
#         (edge_points[i][0][0], edge_points[i][0][1]))
#
# # find all disconnected graphs
#
#
# def dfs(graph, start, visited=None):
#     if visited is None:
#         visited = set()
#     visited.add(start)
#     for next in graph[start] - visited:
#         dfs(graph, next, visited)
#     return visited
#
#
# def connected_components(graph):
#     components = []
#     visited = set()
#     for node in graph:
#         if node not in visited:
#             connected = dfs(graph, node)
#             components.append(connected)
#             visited.update(connected)
#     return components
#
#
# components = connected_components(graph)
# # reshape each component to a numpy array for graham_scan
#
# components = [np.array(list(component)) for component in components]
#
# # reshape to (n, 2)
#
# components = [component.reshape(-1, 2) for component in components]
# # find hull for each disconnected graph
# hulls = []
# for component in components:
#     hull = graham_scan(component, [])
#     hulls.append(hull)
#
# # plot hulls
# for hull_id, hull in enumerate(hulls):
#     component = components[hull_id]
#     for i in range(len(hull[1])-1):
#         plt.add_trace(go.Scatter(x=(component[hull[1][i]][0], component[hull[1][i+1]][0]), y=(
#             component[hull[1][i]][1], component[hull[1][i+1]][1]), mode='lines'), row=2, col=2)


tri = Delaunay(LD_data)

fig, axs = plt.subplots(1, 2)
axs[0].triplot(LD_data[:, 0], LD_data[:, 1], tri.simplices, 'b-')
for i in range(len(edge_points)):
    # plt.add_trace(go.Scatter(x=(edge_points[i][0][0], edge_points[i][1][0]), y=(
    #     edge_points[i][0][1], edge_points[i][1][1]), mode='lines'), row=1, col=2)
    axs[1].plot([edge_points[i][0][0], edge_points[i][1][0]], [
        edge_points[i][0][1], edge_points[i][1][1]], 'b')
axs[0].title.set_text('Delaunay triangulation')
axs[1].title.set_text('Alpha shape')
plt.show()
