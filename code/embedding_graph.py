
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors
from Utils import *


def show_graph(ax, ax2, folder):
    """
    Plots the reduced data and alpha shape edges for a given folder.

    Parameters:
    - ax: The matplotlib axis to draw the plot on.
    - ax2: The matplotlib axis to draw the alpha shape edges on.
    - folder: The folder containing the data files.
    """
    # Load the reduced data
    X_reduced = np.load('./'+folder+'/LD_data.npy')
    ax.plot(X_reduced[:, 0], X_reduced[:, 1], 'o',
            markersize=2, label=f'{folder} Data')

    # Normalize the data for alpha shape computation
    X_reduced_reshaped = (X_reduced - np.min(X_reduced)) / \
        (np.max(X_reduced) - np.min(X_reduced)) * 100

    # Compute alpha shape edges
    _, _, edges = alpha_shape(X_reduced_reshaped, alpha=0.00001)

    # Plot the edges
    for edge in edges:
        ax2.plot(
            [X_reduced[edge[0]][0], X_reduced[edge[1]][0]],
            [X_reduced[edge[0]][1], X_reduced[edge[1]][1]],
            c='black', linewidth=0.5
        )
    ax.legend()
    ax.set_title(f'{folder} Graph')


# List of folders to plot
# folders = ['t-SNE_MNIST_data', 'PCA_MNIST_data',
#            'UMAP_MNIST_data', 'Isomap_MNIST_data', 'MDS_MNIST_data']
# COIL-20 folders
folders = ['t-SNE_COIL-20_data', 'PCA_COIL-20_data',
           'UMAP_COIL-20_data', 'Isomap_COIL-20_data', 'MDS_COIL-20_data']

# Create a figure and axes for subplots
fig, axes = plt.subplots(2, len(folders), figsize=(
    15, 10))

for i, folder in enumerate(folders):
    show_graph(axes[0, i], axes[1, i], folder)


# Adjust layout
plt.tight_layout()
plt.show()
