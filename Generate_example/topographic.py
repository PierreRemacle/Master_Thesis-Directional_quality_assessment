from sklearn.manifold import TSNE
from somperf.utils.topology import rectangular_topology_dist
from sklearn.manifold import LocallyLinearEmbedding as lle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_s_curve
from minisom import MiniSom
from sklearn.metrics.pairwise import euclidean_distances
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
import pandas as pd
import matplotlib.gridspec as gridspec


def topographic_product(dist_fun, som):
    """Topographic product.

    Parameters
    ----------
    dist_fun : function (k : int, l : int) => int
        distance function between units k and l on the map.
    som : array, shape = [n_units, dim]
        SOM code vectors.

    Returns
    -------
    tp : float
        topographic product (tp < 0 when the map is too small, tp > 0 if it is too large)

    References
    ----------
    Bauer, H. U., & Pawelzik, K. R. (1992). Quantifying the Neighborhood Preservation of Self-Organizing Feature Maps.
    """
    n_units = som.shape[0]
    print(n_units)
    original_d = euclidean_distances(som) + 1e-16
    print(som.shape)
    original_knn = np.argsort(original_d, axis=1)
    map_d = np.array([[dist_fun(j, k)
                      for k in range(n_units)]
                      for j in range(n_units)]) + 1e-16
    map_knn = np.argsort(map_d, axis=1)
    # compute Q1 (n_units x n_units-1 matrix)
    q1 = np.array([[np.divide(original_d[j, map_knn[j, k]], original_d[j, original_knn[j, k]])
                   for k in range(1, n_units)]
                   for j in range(n_units)])
    # compute Q2 (n_units x n_units-1 matrix)
    q2 = np.array([[np.divide(map_d[j, map_knn[j, k]], map_d[j, original_knn[j, k]])
                   for k in range(1, n_units)]
                   for j in range(n_units)])
    # compute P3 (n_units x n_units-1 matrix)
    p3 = np.array([[np.prod([(q1[j, l] * q2[j, l])**(1/(2*k)) for l in range(k)])
                   for k in range(1, n_units)]
                   for j in range(n_units)])
    # combine final result (float)
    return np.sum(np.log(p3)) / (n_units * (n_units - 1))


# Generate S-curve data
n_samples = 5000
X, _ = make_s_curve(n_samples=n_samples, random_state=21)

grid_size = int(np.sqrt(5 * np.sqrt(n_samples)))
# Create and train SOM
som = MiniSom(grid_size, grid_size, 3, sigma=1.0, learning_rate=1, activation_distance='euclidean',
              topology='rectangular', neighborhood_function='gaussian', random_seed=21)
initial_weights = som.get_weights().copy().reshape(-1, 3)
som.train(X, 500000, random_order=False, verbose=True)

# Get the weights of the SOM
weights = som.get_weights().reshape(-1, 3)  # Reshape for easier plotting

# Calculate the topological topological_product
map_size = (grid_size, grid_size)
topological_product_value = topographic_product(
    rectangular_topology_dist(map_size), weights)
print("Topological product value: ", topological_product_value)

# Prepare grid coordinates for SOM
grid_x, grid_y = np.meshgrid(np.arange(grid_size), np.arange(grid_size))
grid_coords = np.column_stack([grid_x.flatten(), grid_y.flatten()])

# Display SOM and 3D S-curve on subplots
fig = plt.figure(figsize=(12, 6))
gs = gridspec.GridSpec(2, 6, figure=fig)
# 3D S-Curve plot
ax1 = fig.add_subplot(gs[0, 0], projection='3d')
ax1.scatter(X[:, 0], X[:, 1], X[:, 2], c=X[:, 2], cmap='viridis')

# Elevation and azimuthal angle
ax1.view_init(elev=5.465610421974429, azim=-62.64216503147967)
ax1.set_title('S-Curve')
ax1.set_xlabel('X-axis')
ax1.set_ylabel('Y-axis')
ax1.set_zlabel('Z-axis')

ax3 = fig.add_subplot(gs[1, 0], projection='3d')
ax3.view_init(elev=5.465610421974429, azim=-62.64216503147967)
# Plot SOM weights on top of the S-curve
ax3.scatter(weights[:, 0], weights[:, 1], weights[:, 2],
            c='black', marker='o', s=50, label='SOM Weights')

# Draw grid lines connecting SOM weights
for i in range(grid_size):
    for j in range(grid_size):
        # Connect to right neighbor
        if i < grid_size - 1:
            ax3.plot([weights[i * grid_size + j, 0], weights[(i + 1) * grid_size + j, 0]],
                     [weights[i * grid_size + j, 1],
                         weights[(i + 1) * grid_size + j, 1]],
                     [weights[i * grid_size + j, 2], weights[(i + 1) * grid_size + j, 2]], c='gray', alpha=0.5)
        # Connect to bottom neighbor
        if j < grid_size - 1:
            ax3.plot([weights[i * grid_size + j, 0], weights[i * grid_size + (j + 1), 0]],
                     [weights[i * grid_size + j, 1],
                         weights[i * grid_size + (j + 1), 1]],
                     [weights[i * grid_size + j, 2], weights[i * grid_size + (j + 1), 2]], c='gray', alpha=0.5)

ax3.legend()


# Draw grid lines connecting initial weights

ax3.set_title('SOM Weights')
ax3.set_xlabel('SOM X-axis')
ax3.set_ylabel('SOM Y-axis')


# 2D SOM plot for visualization
ax2 = fig.add_subplot(gs[0:2, 1])
# Create a color map for the SOM
# Create a 10x10 grid for RGB colors
som_data = np.zeros((grid_size, grid_size))
for i in range(grid_size):
    for j in range(grid_size):
        # Assign weights to the corresponding grid
        som_data[i, j] = weights[i * grid_size + j][2]

# Display the SOM weights as an image
ax2.imshow(som_data, origin='lower')
ax2.text(0, -0.5, 'TP: {:.5f}'.format(
    topological_product_value), fontsize=12, transform=plt.gca().transAxes)
ax2.set_title('SOM grid ')
ax2.set_xlabel('SOM X-axis')
ax2.set_ylabel('SOM Y-axis')

# hessian eigenmap
lle_hessian = lle(method="hessian", n_components=2, n_neighbors=12, n_jobs=-1)
S_hessian = lle_hessian.fit_transform(X)
# rescale to 0-1
S_hessian = (S_hessian - S_hessian.min(axis=0)) / \
    (S_hessian.max(axis=0) - S_hessian.min(axis=0))

# Create a grid for RGB colors
hessian_data = np.zeros((grid_size, grid_size))
hessian_occurences = np.zeros((grid_size, grid_size))
hesssian_3D_data = np.zeros((grid_size, grid_size, 3))
for i in range(n_samples):
    point = X[i]
    LD_point = S_hessian[i]
    # coords
    x, y = int(LD_point[1]*grid_size), int(LD_point[0]*grid_size)
    if x == grid_size:
        x = grid_size-1
    if y == grid_size:
        y = grid_size-1

    hessian_data[x, y] += point[2]
    hesssian_3D_data[x, y] += point
    hessian_occurences[x, y] += 1

for i in range(grid_size):
    for j in range(grid_size):
        if hessian_occurences[i, j] != 0:
            hessian_data[i, j] /= hessian_occurences[i, j]
            hesssian_3D_data[i, j] /= hessian_occurences[i, j]

topological_product_value = topographic_product(
    rectangular_topology_dist(map_size), hesssian_3D_data.reshape(-1, 3))
print("Topological product value: ", topological_product_value)

ax4 = fig.add_subplot(gs[0:2, 2])
ax4.scatter(S_hessian[:, 0], S_hessian[:, 1], c=X[:, 2], cmap='viridis')
# plot a grid on top of the scatter plot

for i in range(grid_size+1):
    ax4.axvline(x=i/grid_size, color='gray', alpha=0.5)
    ax4.axhline(y=i/grid_size, color='gray', alpha=0.5)


# ax4.imshow(hessian_data, cmap='viridis', interpolation='none')
ax4.set_title('Hessian Eigenmap')
ax4.set_xlabel('Hessian X-axis')
ax4.set_ylabel('Hessian Y-axis')

ax5 = fig.add_subplot(gs[0:2, 3])
ax5.imshow(hessian_data, origin='lower')
ax5.set_title('Hessian Eigenmap')
ax5.set_xlabel('Hessian X-axis')
ax5.set_ylabel('Hessian Y-axis')
ax5.text(0, -0.5, 'TP: {:.5f}'.format(
    topological_product_value), fontsize=12, transform=plt.gca().transAxes)

# tsne

embedding_TSNE = TSNE(n_components=2)
X_transformed_TSNE = embedding_TSNE.fit_transform(X)

# rescale to 0-1
X_transformed_TSNE = (X_transformed_TSNE - X_transformed_TSNE.min(axis=0)) / \
    (X_transformed_TSNE.max(axis=0) - X_transformed_TSNE.min(axis=0))

# create a grid for RGB colors
tsne_data = np.zeros((grid_size, grid_size))
tsne_occurences = np.zeros((grid_size, grid_size))
tsne_3D_data = np.zeros((grid_size, grid_size, 3))

for i in range(n_samples):
    point = X[i]
    LD_point = X_transformed_TSNE[i]
    # coords
    x, y = int(LD_point[1]*grid_size), int(LD_point[0]*grid_size)
    if x == grid_size:
        x = grid_size-1
    if y == grid_size:
        y = grid_size-1

    tsne_data[x, y] += point[2]
    tsne_3D_data[x, y] += point
    tsne_occurences[x, y] += 1

for i in range(grid_size):
    for j in range(grid_size):
        if tsne_occurences[i, j] != 0:
            tsne_data[i, j] /= tsne_occurences[i, j]
            tsne_3D_data[i, j] /= tsne_occurences[i, j]

topological_product_value = topographic_product(
    rectangular_topology_dist(map_size), tsne_3D_data.reshape(-1, 3))
print("Topological product value: ", topological_product_value)

ax6 = fig.add_subplot(gs[0:2, 4])
ax6.scatter(X_transformed_TSNE[:, 0],
            X_transformed_TSNE[:, 1], c=X[:, 2], cmap='viridis')
# plot a grid on top of the scatter plotting
for i in range(grid_size+1):
    ax6.axvline(x=i/grid_size, color='gray', alpha=0.5)
    ax6.axhline(y=i/grid_size, color='gray', alpha=0.5)

# ax6.imshow(tsne_data, cmap='viridis', interpolation='none')
ax6.set_title('t-SNE')
ax6.set_xlabel('t-SNE X-axis')
ax6.set_ylabel('t-SNE Y-axis')

ax7 = fig.add_subplot(gs[0:2, 5])
ax7.imshow(tsne_data, origin='lower')
ax7.set_title('t-SNE')
ax7.set_xlabel('t-SNE X-axis')
ax7.set_ylabel('t-SNE Y-axis')
ax7.text(0, -0.5, 'TP: {:.2f}'.format(
    topological_product_value), fontsize=12, transform=plt.gca().transAxes)


plt.tight_layout()
plt.savefig('Topological_product_SCurve.svg', format='svg')
plt.show()
