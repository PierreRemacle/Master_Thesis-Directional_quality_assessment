# compute_rnx.py
import os
from PIL import Image


import numpy as np
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE, Isomap, MDS
import umap
from minisom import MiniSom  # SOM implementation
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

# Define a function to compute RNX for a given dimensionality reduction method


def compute_rnx(X, X_reduced, n_neighbors):
    # Compute the original and reduced nearest neighbors
    neighbors_original = NearestNeighbors(n_neighbors=n_neighbors).fit(X)
    neighbors_reduced = NearestNeighbors(
        n_neighbors=n_neighbors).fit(X_reduced)

    original_indices = neighbors_original.kneighbors(return_distance=False)
    reduced_indices = neighbors_reduced.kneighbors(return_distance=False)

    # Compute the RNX score by comparing neighborhood preservation
    intersection_counts = [
        len(set(original_indices[i]) & set(reduced_indices[i])) for i in range(X.shape[0])
    ]
    qnx = np.mean(intersection_counts) / n_neighbors
    rnx = (((X.shape[0] - 1) * qnx) - n_neighbors) / \
        (X.shape[0] - 1 - n_neighbors)
    return rnx

# Load MNIST and COIL-20 datasets


def load_datasets():
    # Load MNIST
    mnist = datasets.fetch_openml('mnist_784')
    # only 3000 samples are used for performance reasons
    mnist.data = mnist.data[:3000]
    mnist.target = mnist.target[:3000]
    X_mnist = mnist.data / 255.0  # Normalize
    y_mnist = mnist.target.astype(int)

    # Load COIL-20 (assuming it’s available or downloaded manually)
    # Replace with actual COIL-20 loading code or path if using local files
    # For simplicity, we assume coil20 is a numpy array with shape (N, D) and labels y_coil
    X_coil, y_coil = load_coil20()  # Placeholder function

    return (X_mnist, y_mnist), (X_coil, y_coil)

# Placeholder function for loading COIL-20 data


def load_coil20():
    # Load COIL-20 data from a local source or similar
    # This should be replaced by the actual loading method for COIL-20

    path = "../DATASETS/coil-20/"
    folders = os.listdir(path)
    folders.remove(".DS_Store")
    out = []
    for i, folder in enumerate(folders):
        files = os.listdir(path + folder)
        for image in files:
            img = Image.open(path + folder + "/" + image)
            img = np.asarray(img)
            img = img.flatten()
            out.append(img)
    print(len(out))
    return np.array(out[:3000]), np.array(folders)


def apply_reduction_methods(X, y):
    # Standardize features
    X = StandardScaler().fit_transform(X)

    methods = {
        "PCA": PCA(n_components=2),
        "t-SNE": TSNE(n_components=2, random_state=42, max_iter=300),
        "UMAP": umap.UMAP(n_components=2),
        "Isomap": Isomap(n_components=2),
        "MDS": MDS(n_components=2),
        "SOM": None  # Placeholder for SOM, since it needs a separate implementation
    }

    results = {}
    for name, method in methods.items():
        print(f"Applying {name}...")
        if name == "SOM":
            som = MiniSom(10, 10, X.shape[1], sigma=0.5, learning_rate=0.5)
            som.train_random(X, 100)
            X_reduced = np.array([som.winner(x) for x in X]).reshape(-1, 2)
        # LDA needs labels and >=2 classes
        elif name == "LDA" and len(np.unique(y)) > 1:
            X_reduced = method.fit_transform(X, y)
        else:
            X_reduced = method.fit_transform(X)

        results[name] = X_reduced
    return results

# Run RNX for all methods and datasets


def main():
    # Load datasets
    (X_mnist, y_mnist), (X_coil, y_coil) = load_datasets()

    rnx_results_mnist = {}
    rnx_results_coil = {}

    # Compute RNX for each neighborhood size and each dataset
    for dataset_name, (X, y) in [("COIL-20", (X_coil, y_coil)), ("MNIST", (X_mnist, y_mnist))]:
        if dataset_name == "COIL-20":
            neighborhood_sizes = range(30, 1440, 30)
        else:  # MNIST
            neighborhood_sizes = range(30, 3000, 30)
        print(f"Results for {dataset_name} dataset:")
        rnx_results = {}

        # Apply each method and calculate RNX for different neighborhood sizes
        reductions = apply_reduction_methods(X, y)
        for method_name, X_reduced in reductions.items():
            rnx_results[method_name] = []
            for n_neighbors in neighborhood_sizes:
                print(n_neighbors)
                rnx_score = compute_rnx(X, X_reduced, n_neighbors)
                rnx_results[method_name].append(rnx_score)

        # Store results for plotting
        if dataset_name == "MNIST":
            rnx_results_mnist = rnx_results
        else:
            rnx_results_coil = rnx_results

        if not os.path.exists('../PLOTS'):
            os.makedirs('../PLOTS')
        # Plot RNX for this dataset
        plt.figure(figsize=(10, 6))
        for method_name, rnx_scores in rnx_results.items():
            np.savetxt(f"../PLOTS/{dataset_name}_{method_name}_rnx.csv",
                       np.array(rnx_scores).T, delimiter=",")
            plt.plot(neighborhood_sizes, rnx_scores,
                     marker='o', label=method_name)

        plt.title(f'RNX Scores for {dataset_name} Dataset')
        plt.xlabel('Size of Neighborhood')
        plt.ylabel('RNX Score')
        plt.xscale('log')
        plt.grid(True)
        plt.legend()
        plt.show()


if __name__ == "__main__":
    main()
