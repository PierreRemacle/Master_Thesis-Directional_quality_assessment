from ipywidgets import interact, IntSlider
from mpl_toolkits.mplot3d import art3d
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import MDS
from Utils import *
import sys
import numpy
import matplotlib.pyplot as pyplot
import matplotlib as matplotlib
from alive_progress import alive_bar
import sklearn
import sklearn.datasets

from matplotlib.widgets import Slider

s_curve = sklearn.datasets.make_s_curve(
    n_samples=1000, noise=0.0, random_state=0)

# plot s curve in 2d using mulidimensional scaling

mds = MDS(n_components=2, random_state=0)
s_curve_2d = mds.fit_transform(s_curve[0])
# rescale the data to be between 0 and 100
s_curve_2d_reshaped = (s_curve_2d - np.min(s_curve_2d)) / \
    (np.max(s_curve_2d) - np.min(s_curve_2d)) * 100
# save the data to a file
df_2d = pd.DataFrame(s_curve_2d)
df_2d.to_csv("../DATA/s_curve_LD.csv")
df_3d = pd.DataFrame(s_curve[0])
df_3d.to_csv("../DATA/s_curve_HD.csv")

_, _, edges = alpha_shape(s_curve_2d_reshaped, alpha=0.3)
# display edges
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(s_curve_2d[:, 0], s_curve_2d[:, 1])
for edge in edges:
    edge = list(edge)
    ax.plot([s_curve_2d[edge[0]][0], s_curve_2d[edge[1]][0]], [
            s_curve_2d[edge[0]][1], s_curve_2d[edge[1]][1]], c='r')
plt.show()
distancmatrix3D = compute_distance_matrix(s_curve[0])
distance_matrix_LD = compute_distance_matrix(s_curve_2d)
graph = distance_matrix_LD.copy()
graph = graph**2 + np.max(graph) * 100
for i in range(len(graph)):
    graph[i][i] = 0  # remove self loop
for edge in edges:
    graph[edge[0]][edge[1]] = distance_matrix_LD[edge[0]][edge[1]]**2
    graph[edge[1]][edge[0]] = distance_matrix_LD[edge[1]][edge[0]]**2
print(graph)


def update_plot(point=99, number_of_neighbors=512):
    # Find the neighbors of the point
    neighbors = []
    distance_to_point = []
    for i in range(len(s_curve_2d)):
        neighbors.append(i)
        distance_to_point.append(np.linalg.norm(
            s_curve_2d[point] - s_curve_2d[i]))

    # Sort based on distance 2D
    neighbors = np.array(neighbors)
    distance_to_point = np.array(distance_to_point)
    neighbors = neighbors[np.argsort(distance_to_point)]
    distance_to_point = distance_to_point[np.argsort(distance_to_point)]
    radius_2D = distance_to_point[number_of_neighbors]

    # Sort based on distance 3D
    neighbors3D = []
    distance_to_point = []
    for i in range(len(s_curve[0])):
        neighbors3D.append(i)
        distance_to_point.append(np.linalg.norm(
            s_curve[0][point] - s_curve[0][i]))

    neighbors3D = np.array(neighbors3D)
    distance_to_point = np.array(distance_to_point)
    neighbors3D = neighbors3D[np.argsort(distance_to_point)]
    distance_to_point = distance_to_point[np.argsort(distance_to_point)]
    radius_3D = distance_to_point[number_of_neighbors]

    colors2D = []
    for i in range(len(s_curve_2d)):
        if i == point:
            colors2D.append('r')
        elif i == neighbors[number_of_neighbors]:
            colors2D.append('g')
        elif i in neighbors[:number_of_neighbors]:
            colors2D.append('orange')
        else:
            colors2D.append('b')
    colors3D = []
    for i in range(len(s_curve[0])):
        if i == point:
            colors3D.append('r')

        elif i in neighbors3D[:number_of_neighbors]:
            colors3D.append('orange')
        else:
            colors3D.append('b')

    # Clear previous plots
    ax1.clear()
    ax2.clear()
    ax3.clear()
    ax4.clear()

    # plot 2D in the orders of the corlors , with the red point first
    ax1.scatter(s_curve_2d[:, 0], s_curve_2d[:, 1], c=colors2D)
    ax1.scatter(s_curve_2d[point][0], s_curve_2d[point][1], c='r')
    ax1.scatter(s_curve_2d[neighbors[number_of_neighbors]][0],
                s_curve_2d[neighbors[number_of_neighbors]][1], c='g')
    # plot circle
    circle = plt.Circle(s_curve_2d[point], radius_2D, color='r', fill=False)
    ax1.add_artist(circle)
    ax1.set_title('2D')
    ax1.set_aspect("equal")

    # plot 3D
    ax2.scatter(s_curve[0][:, 0], s_curve[0][:, 1],
                s_curve[0][:, 2], c=colors3D)
    # wireframe sphere
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = radius_3D*np.cos(u)*np.sin(v) + s_curve[0][point][0]
    y = radius_3D*np.sin(u)*np.sin(v) + s_curve[0][point][1]
    z = radius_3D*np.cos(v) + s_curve[0][point][2]
    ax2.plot_wireframe(x, y, z, color="r", alpha=0.2)
    # axes have same scale
    ax2.set_box_aspect([1, 1, 1])
    ax2.set_aspect("equal")
    ax2.set_title('3D')
    # find path from point to point

    LD_path = list(BasicAStar(graph, distance_matrix_LD,
                   LD_data).astar(point, neighbors[number_of_neighbors]))
    HD_path = create_HD_path2(LD_path, distancmatrix3D,
                              point, neighbors[number_of_neighbors])
    HD_path = np.array(HD_path, dtype=int)
    ax3.scatter(s_curve_2d[:, 0], s_curve_2d[:, 1], c="b")
    ax4.scatter(s_curve[0][:, 0], s_curve[0][:, 1],
                s_curve[0][:, 2], c="b", alpha=0.5, linewidths=0.)
    ax3.scatter(s_curve_2d[point][0], s_curve_2d[point][1], c='r')
    ax3.scatter(s_curve_2d[neighbors[number_of_neighbors]][0],
                s_curve_2d[neighbors[number_of_neighbors]][1], c='g')
    ax4.scatter(s_curve[0][point][0], s_curve[0][point]
                [1], s_curve[0][point][2], c='r')
    ax4.scatter(s_curve[0][neighbors[number_of_neighbors]][0], s_curve[0]
                [neighbors[number_of_neighbors]][1], s_curve[0][neighbors[number_of_neighbors]][2], c='g')

    for i in range(len(LD_path)-1):
        ax3.plot([s_curve_2d[LD_path[i]][0], s_curve_2d[LD_path[i+1]][0]],
                 [s_curve_2d[LD_path[i]][1], s_curve_2d[LD_path[i+1]][1]], c='r')
    for i in range(len(HD_path)-1):
        ax4.plot([s_curve[0][HD_path[i]][0], s_curve[0][HD_path[i+1]][0]],
                 [s_curve[0][HD_path[i]][1], s_curve[0][HD_path[i+1]][1]],
                 [s_curve[0][HD_path[i]][2], s_curve[0][HD_path[i+1]][2]], c='r')
    ax3.set_title('2D')
    ax4.set_title('3D')
    ax3.set_aspect("equal")
    ax4.set_box_aspect([1, 1, 1])
    ax4.set_aspect("equal")
    intersec = np.intersect1d(neighbors, neighbors3D)
    print("intersection", len(intersec)/len(neighbors))
    print("editdistance", levenshteinDistanceDP(LD_path, HD_path))

    # Update the canvas
    plt.draw()


# Create initial plot
fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222, projection='3d')
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224, projection='3d')
update_plot()
# Create sliders
ax_point = plt.axes([0.1, 0.01, 0.65, 0.03])
ax_neighbors = plt.axes([0.1, 0.05, 0.65, 0.03])


slider_point = Slider(ax_point, 'point', 0, len(
    s_curve_2d)-1, valinit=99, valstep=1)
slider_neighbors = Slider(ax_neighbors, 'number_of_Neighbors', 1, len(
    s_curve_2d)-1, valinit=512, valstep=1)

# Register callback function
slider_point.on_changed(lambda val: update_plot(
    int(val), int(slider_neighbors.val)))
slider_neighbors.on_changed(
    lambda val: update_plot(int(slider_point.val), int(val)))
# Show plot
plt.show()
