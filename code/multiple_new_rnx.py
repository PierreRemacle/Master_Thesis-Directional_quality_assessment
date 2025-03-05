
# compute_rnx.py
from concurrent.futures import ThreadPoolExecutor
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
from Utils import *

# Define a function to compute RNX for a given dimensionality reduction method


def compute_new_rnx(X, X_reduced, embedding_name):

    X_reduced_reshaped = (X_reduced - np.min(X_reduced)) / \
        (np.max(X_reduced) - np.min(X_reduced)) * 100
    # distord the data to see the impact of the distortion
    # multiply the x coordinates by 1.5
    X_reduced_reshaped[:, 0] = X_reduced_reshaped[:, 0]
    _, _, edges = alpha_shape(X_reduced_reshaped, alpha=0.00001)

    # results, localisation_of_errors, error_per_start = random_selection_path_2(
    #     X, X_reduced, 200, graph)
    # xss, yss, index_enveloppes = enveloppe_of_cluster(X_reduced)
    # flatten the list
    # index_enveloppe = [
    #     item for sublist in index_enveloppes for item in sublist]
    # results, localisation_of_errors, error_per_start = ALL_path_enveloppe_HDcompare(
    #     X, X_reduced, graph, index_enveloppe)
    ALL_path_3(
        X, X_reduced, edges, embedding_name)
    # average = {}
    # for key,key2 in results:
    #     average[key] = sum(results[key]) / (len(results[key]))
    # average = dict(sorted(average.items()))
    #
    # x = np.sort(np.array(list(average.keys())))
    # y_values = np.array(list(average.values()))
    # x_steps = np.asarray(list(average.keys()))
    # y_diago = (x_steps-2)/(x_steps)
    # standerized_x = (x_steps-2) / np.max(x_steps-2) * 100
    # my_rnx = 1-(y_values / y_diago)
    # return x, my_rnx


def load_datasets():
    # Load MNIST
    mnist = datasets.fetch_openml('mnist_784')
    # only 3000 samples are used for performance reasons
    mnist.data = mnist.data[:3000]
    mnist.target = mnist.target[:3000]
    X_mnist = mnist.data / 255.0  # Normalize
    X_mnist = np.array(X_mnist, dtype=float)
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
    return np.array(out), np.array(folders)


def apply_reduction_methods(X, y, method_name):
    # Standardize features
    X = StandardScaler().fit_transform(X)

    methods = {
        "PCA": PCA(n_components=2),
        "t-SNE": TSNE(n_components=2, random_state=42, max_iter=300),
        "UMAP": umap.UMAP(n_components=2),
        "Isomap": Isomap(n_components=2),
        "MDS": MDS(n_components=2),
    }

    method = methods[method_name]
    name = method_name
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

    return X_reduced

# Run RNX for all methods and datasets


def main():
    # Load datasets
    (X_mnist, y_mnist), (X_coil, y_coil) = load_datasets()

    rnx_results_mnist = {}
    rnx_results_coil = {}

    # Compute RNX for each neighborhood size and each dataset
    # for dataset_name, (X, y) in [("MNIST", (X_mnist, y_mnist)), ("COIL-20", (X_coil, y_coil))]:
    for dataset_name, (X, y) in [("MNIST", (X_mnist, y_mnist))]:
        # for dataset_name, (X, y) in [("COIL-20", (X_coil, y_coil))]:

        print(f"Results for {dataset_name} dataset:")
        rnx_results = {}
        x_results = {}
        # Apply each method and calculate RNX for different neighborhood sizes
        for method_name in ["PCA", "t-SNE"]:

            X_reduced = apply_reduction_methods(X, y, method_name)
            compute_new_rnx(X, X_reduced, method_name + "_" + dataset_name)

            # rnx_results[method_name] = y
            # x_results[method_name] = x

        # Store results for plotting
        # if dataset_name == "MNIST":
        #     rnx_results_mnist = rnx_results
        # else:
        #     rnx_results_coil = rnx_results

        # # Plot RNX for this dataset
        # plt.figure(figsize=(10, 6))
        # for method_name, rnx_scores in rnx_results.items():
        #     # save to a specific folder, if needed create the file
        #     if not os.path.exists('../PLOTS'):
        #         os.makedirs('../PLOTS')
        #
        #     np.savetxt(f"../PLOTS/{dataset_name}_{method_name}_new_rnx_all_2.csv",
        #                np.array(rnx_scores).T, delimiter=",")
        #     plt.plot(x_results[method_name], rnx_scores,
        #              marker='o', label=method_name)
        #
        # plt.title(f'RNX Scores for {dataset_name} Dataset')
        # plt.xlabel('Size of Neighborhood')
        # plt.ylabel('RNX Score')
        # plt.xscale('log')
        # plt.grid(True)
        # plt.legend()
        # plt.show()


if __name__ == "__main__":
    main()
