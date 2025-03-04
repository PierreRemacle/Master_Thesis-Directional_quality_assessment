
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.csgraph import shortest_path
from scipy.sparse import csr_matrix
from sklearn.manifold import LocallyLinearEmbedding as lle
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
from sklearn.manifold import MDS
from sklearn.datasets import load_digits
import pandas as pd
from zadu import zadu
from scipy.spatial.distance import pdist, squareform
from sklearn.datasets import make_s_curve
from mpl_toolkits.mplot3d import Axes3D

from scipy.spatial import Delaunay


X, Y = make_s_curve(n_samples=1000, random_state=21)

# Hessian eigenmap
lle_hessian = lle(method="hessian", n_components=2, n_neighbors=12, n_jobs=-1)
S_hessian = lle_hessian.fit_transform(X)
S_hessian = (S_hessian - S_hessian.min(axis=0)) / \
    (S_hessian.max(axis=0) - S_hessian.min(axis=0))

# select a start point
idx = 120

# select a end point

idx_end = 300

# Delaunay on S_hessian

tri = Delaunay(S_hessian)
# find the shortest path


def delaunay_graph(LD_data):
    tri = Delaunay(LD_data)
    grah = np.zeros((len(LD_data), len(LD_data)))
    for i in range(len(tri.simplices)):
        for j in range(3):
            grah[tri.simplices[i][j]][tri.simplices[i]
                                      [(j+1) % 3]] = np.linalg.norm(LD_data[tri.simplices[i][j]] - LD_data[tri.simplices[i][(j+1) % 3]])**2
            grah[tri.simplices[i][(j+1) % 3]][tri.simplices[i][j]
                                              ] = grah[tri.simplices[i][j]][tri.simplices[i][(j+1) % 3]]
    return grah


# create a graph
graph = delaunay_graph(S_hessian)

# find the shortest path with a star
graph = csr_matrix(graph)
dist_matrix, predecessors = shortest_path(
    csgraph=graph, directed=False, indices=idx, return_predecessors=True)
path = []
while idx_end != idx:
    path.append(idx_end)
    idx_end = predecessors[idx_end]
path.append(idx)
path = path[::-1]

HD_path = []
distance = []
for i in path:
    print(X[i])
    print(X[idx])
    distance.append(np.linalg.norm(X[idx] - X[i]))

sorted_distance = np.argsort(distance)
# reverse the list

for i in sorted_distance:
    HD_path.append(path[i])
print(HD_path)

# Co-ranking Matrix

co_ranking_matrix = np.zeros((len(path), len(path)))
for i in range(len(path)-1):
    path_element = path[i]
    indice_in_HD_path = HD_path.index(path_element)
    print(path_element, indice_in_HD_path)
    co_ranking_matrix[i, indice_in_HD_path] = 1


def QNX(LD_DATA, HD_DATA, k):
    """ Average Normalized Agreement Between K-ary Neighborhoods (QNX)"""
    knn1 = NearestNeighbors(n_neighbors=k)
    knn1.fit(LD_DATA)
    knn2 = NearestNeighbors(n_neighbors=k)
    knn2.fit(HD_DATA)
    intersections = []
    for i in range(len(LD_DATA)):
        LD_indices = knn1.kneighbors(
            [LD_DATA[i]], k, return_distance=False)
        HD_indices = knn2.kneighbors(
            [HD_DATA[i]], k, return_distance=False)
        intersection = np.intersect1d(LD_indices, HD_indices)
        intersections.append(len(intersection))
    return sum(intersections) / ((len(LD_DATA)) * (k))


def RNX(LD_data, HD_data, k):
    """ Rescaled Agreement Between K-ary Neighborhoods (RNX)"""
    N = len(LD_data)
    qnx = QNX(LD_data, HD_data, k)

    return (((N-1) * qnx) - k) / (N - 1 - k)


def RNX2(K, co_ranking_matrix):
    UN = 0
    UX = 0
    UP = 0

    for i in range(K):
        for j in range(K):
            if i > j:
                UN += co_ranking_matrix[i, j]
            if i < j:
                UX += co_ranking_matrix[i, j]
            if i == j:
                UP += co_ranking_matrix[i, j]
    UN = UN / (K)
    UX = UX / (K)
    UP = UP / (K)
    Q = UN + UX + UP
    R = ((len(X)-1) * Q - K) / (len(X) - 1 - K)
    return R


all_RNX = []
all_RNX2 = []
k_values = range(1, len(X), 20)  # Values for k from 1 to max_k
for k in k_values:
    all_RNX.append(RNX(X, S_hessian, k))
print(all_RNX)

# LCMC

scores = []
for k in k_values:
    print(f"Computing LCMC for with k={k}")

    # Define the specification for the zadu object
    spec_list = [
        {
            "id": "lcmc",
            "params": {
                "k": k
            }
        }
    ]

    # Create a ZADU object with the current specification for digits
    zadu_digits = zadu.ZADU(spec_list, X, return_local=True)
    scores_digits, _ = zadu_digits.measure(S_hessian, X)

    # Store the LCMC score for digits
    local_lcmc_digits = scores_digits[0]["lcmc"]
    Q = local_lcmc_digits + ((k)/(len(X)-1))
    RNX = ((len(X)-1) * Q - k) / (len(X) - 1 - k)

    scores.append(RNX)


fig = plt.figure(figsize=(15, 5))
ax1 = fig.add_subplot(231, projection='3d')
ax1.scatter(X[:, 0], X[:, 1], X[:, 2], c=Y, cmap=plt.cm.Spectral)
ax1.scatter(X[idx, 0], X[idx, 1], X[idx, 2], c='r', s=100)
ax1.scatter(X[idx_end, 0], X[idx_end, 1], X[idx_end, 2], c='g', s=100)
for i in range(len(path)-1):
    ax1.plot(X[[path[i], path[i+1]], 0], X[[path[i], path[i+1]], 1],
             X[[path[i], path[i+1]], 2], 'r')
for i in range(len(HD_path)-1):
    ax1.plot(X[[HD_path[i], HD_path[i+1]], 0], X[[HD_path[i], HD_path[i+1]], 1],
             X[[HD_path[i], HD_path[i+1]], 2], 'g')
plt.title("Original S curve")

ax2 = fig.add_subplot(232)
ax2.scatter(S_hessian[:, 0], S_hessian[:, 1], c=Y, cmap=plt.cm.Spectral)
ax2.scatter(S_hessian[idx, 0], S_hessian[idx, 1], c='r', s=100)
ax2.scatter(S_hessian[idx_end, 0], S_hessian[idx_end, 1], c='g', s=100)
for i in range(len(path)-1):
    ax2.plot(S_hessian[[path[i], path[i+1]], 0],
             S_hessian[[path[i], path[i+1]], 1], 'r')
plt.title("Perfect embedding of S curve")


# plot tri on S_hessian
ax3 = fig.add_subplot(233)
ax3.triplot(S_hessian[:, 0], S_hessian[:, 1], tri.simplices)
for i in range(len(path)-1):
    ax3.plot(S_hessian[[path[i], path[i+1]], 0],
             S_hessian[[path[i], path[i+1]], 1], 'r')
# plot the co-ranking matrix
ax4 = fig.add_subplot(234)
ax4.imshow(co_ranking_matrix, cmap='gray')
plt.title("Co-ranking Matrix")

# plot the QNX
ax5 = fig.add_subplot(235)
ax5.plot(k_values, scores)
# ax5.set_xscale('log')
plt.title("comparaison of path HD and LD ")

# plot the LCMC
ax6 = fig.add_subplot(236)
# log scale for the x axis
ax6.plot(k_values, all_RNX)
# ax6.set_xscale('log')  # Set x-axis to log scale


plt.title("RNX")

plt.show()
