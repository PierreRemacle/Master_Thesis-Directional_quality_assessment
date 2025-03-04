from Utils import *

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(LD_data[:, 0], LD_data[:, 1], marker='o', s=5)
# graph = delaunay_graph(LD_data)
# for i in range(len(graph)):
#     for j in range(len(graph[i])):
#         if graph[i][j] != 0:
#             ax.plot([LD_data[i][0], LD_data[j][0]], [
#                     LD_data[i][1], LD_data[j][1]], c='r')
LD_data_rescale = (LD_data - np.min(LD_data)) / \
    (np.max(LD_data) - np.min(LD_data)) * 100
_, _, edges = alpha_shape(LD_data_rescale, alpha=0.3)
for edge in edges:
    edge = list(edge)
    ax.plot([LD_data[edge[0]][0], LD_data[edge[1]][0]], [
            LD_data[edge[0]][1], LD_data[edge[1]][1]], c='r')
plt.show()
