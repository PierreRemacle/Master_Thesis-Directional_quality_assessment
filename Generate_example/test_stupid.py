

from scipy.spatial import Delaunay
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

import shapely.geometry as geometry
import math
from shapely.ops import cascaded_union, polygonize
from sknetwork.clustering import Louvain, get_modularity
from sklearn.cluster import AffinityPropagation
import time
# from numba import jit
import os
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import warnings
from scipy.spatial import Delaunay


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
        print(LD_indices)
        print(HD_indices)
        intersection = np.intersect1d(LD_indices, HD_indices)
        print(len(intersection))
        intersections.append(len(intersection))
    return sum(intersections) / ((len(LD_DATA)) * (k))


def RNX(LD_data, HD_data, k):
    """ Rescaled Agreement Between K-ary Neighborhoods (RNX)"""
    N = len(LD_data)
    qnx = QNX(LD_data, HD_data, k)

    return (((N-1) * qnx) - k) / (N - 1 - k)


X = []
X_LD = []
for i in range(1, 25):
    X.append([i, i, i+i])
    X_LD.append([i, i])

X = np.array(X)
X_LD = np.array(X_LD)
k_values = range(1, len(X)-1, 1)  # Values for k from 1 to max_k
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
    scores_digits, _ = zadu_digits.measure(X_LD, X)

    # Store the LCMC score for digits
    local_lcmc_digits = scores_digits[0]["lcmc"]
    Q = local_lcmc_digits + ((k)/(len(X)-1))
    R = ((len(X)-1) * Q - k) / (len(X) - 1 - k)

    scores.append(local_lcmc_digits)

rs = []
for k in k_values:
    print(f"Computing RNX for with k={k}")
    r = RNX(X_LD, X, k)
    rs.append(r)
# plot X in 3D
fig = plt.figure()
ax = fig.add_subplot(141, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2])

# plot X in 2D
ax = fig.add_subplot(142)
ax.scatter(X_LD[:, 0], X_LD[:, 1])

# plot the LCMC scores
ax = fig.add_subplot(143)
ax.plot(k_values, scores)

# plot the RNX scores
ax = fig.add_subplot(144)
ax.plot(k_values, rs)

plt.show()
