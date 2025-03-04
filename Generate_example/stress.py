from sklearn.manifold import LocallyLinearEmbedding as lle
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
from sklearn.manifold import MDS
from sklearn.datasets import load_digits
import pandas as pd

from scipy.spatial.distance import pdist, squareform
from sklearn.datasets import make_s_curve
from mpl_toolkits.mplot3d import Axes3D


def stress(LD_data, HD_data):
    """
    Calculate the stress value for assessing dimensionality reduction.

    Parameters:
        LD_data (numpy.ndarray): The low-dimensional representation of the data.
        HD_data (numpy.ndarray): The high-dimensional representation of the data.

    Returns:
        float: The stress value indicating the quality of the dimensionality reduction.
    """
    # Calculate pairwise distances in high-dimensional space
    HD_distances = pdist(HD_data)
    HD_distances_squared_sum = np.sum(HD_distances**2)

    # Calculate pairwise distances in low-dimensional space
    LD_distances = pdist(LD_data)

    # Calculate the numerator (sum of squared differences)
    numerator = np.sum((HD_distances - LD_distances)**2)

    # Calculate the stress value
    stress_value = np.sqrt(numerator / HD_distances_squared_sum)

    return stress_value


# S curve with color
X, Y = make_s_curve(n_samples=1000, random_state=21)
X = X[:1000]
Y = Y[:1000]
print(X.shape)
embedding_MDS = MDS(n_components=2, normalized_stress=False)
X_transformed = embedding_MDS.fit_transform(X)
print(stress(X, X_transformed))

# tsne
embedding_TSNE = TSNE(n_components=2)
X_transformed_TSNE = embedding_TSNE.fit_transform(X)
# rescale to fit the range of MDS
MDS_min = X_transformed.min(axis=0)
MDS_max = X_transformed.max(axis=0)
X_transformed_TSNE = (X_transformed_TSNE - X_transformed_TSNE.min(axis=0)) / \
    (X_transformed_TSNE.max(axis=0) - X_transformed_TSNE.min(axis=0))
X_transformed_TSNE = X_transformed_TSNE * (MDS_max - MDS_min) + MDS_min

print(stress(X, X_transformed_TSNE))

# Hessian eigenmap
lle_hessian = lle(method="hessian", n_components=2, n_neighbors=12, n_jobs=-1)
S_hessian = lle_hessian.fit_transform(X)
# rescale to fit the range of MDS
S_hessian = (S_hessian - S_hessian.min(axis=0)) / \
    (S_hessian.max(axis=0) - S_hessian.min(axis=0))
S_hessian = S_hessian * (MDS_max - MDS_min) + MDS_min
print(stress(X, S_hessian))

plt.figure(figsize=(18, 6))

# 3D S-curve plot
ax1 = plt.subplot(1, 4, 1, projection='3d')
ax1.scatter(X[:, 0], X[:, 1], X[:, 2], c=Y)
ax1.set_title('Original S-Curve')
ax1.set_xlabel('X-axis')
ax1.set_ylabel('Y-axis')
ax1.set_zlabel('Z-axis')

# MDS plot
plt.subplot(1, 4, 2)
plt.scatter(X_transformed[:, 0], X_transformed[:, 1], c=Y)
plt.text(
    0.65, 0.01, 'Stress: {:.2f}'.format(stress(X, X_transformed)), fontsize=12, transform=plt.gca().transAxes)
plt.title('MDS')
plt.xlabel('Component 1')
plt.ylabel('Component 2')

# t-SNE plot
plt.subplot(1, 4, 3)
plt.scatter(X_transformed_TSNE[:, 0], X_transformed_TSNE[:, 1], c=Y)
plt.text(0.65, 0.01, 'Stress: {:.2f}'.format(
    stress(X, X_transformed_TSNE)), fontsize=12, transform=plt.gca().transAxes)
plt.title('t-SNE')
plt.xlabel('Component 1')
plt.ylabel('Component 2')

# Hessian eigenmap plot
plt.subplot(1, 4, 4)
plt.scatter(S_hessian[:, 0], S_hessian[:, 1], c=Y)
plt.text(0.65, 0.01, 'Stress: {:.2f}'.format(
    stress(X, S_hessian)), fontsize=12, transform=plt.gca().transAxes)
plt.title('Hessian Eigenmap')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.tight_layout()

plt.savefig('stress_scurve.svg', format='svg')
plt.show()
