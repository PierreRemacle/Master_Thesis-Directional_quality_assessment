# data_cache.py

import numpy as np
import pickle
import plotly.graph_objs as go
import os
from Utils import levenshteinDistanceDP
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors


def compute_rnx(X, X_reduced, steps, folder):
    # Compute the original and reduced nearest neighbors
    rnx_graph = []
    neighbors_original = NearestNeighbors(n_neighbors=len(X_reduced)-1).fit(X)
    neighbors_reduced = NearestNeighbors(
        n_neighbors=len(X_reduced)-1).fit(X_reduced)

    original_indices = neighbors_original.kneighbors(return_distance=False)
    reduced_indices = neighbors_reduced.kneighbors(return_distance=False)
    print(original_indices)

    for n_neighbors in steps:

        print(n_neighbors)

        # Compute the RNX score by comparing neighborhood preservation
        intersection_counts = [
            len(set(original_indices[i][:n_neighbors]) & set(reduced_indices[i][:n_neighbors])) for i in range(X.shape[0])
        ]
        qnx = np.mean(intersection_counts) / n_neighbors
        rnx = (((X.shape[0] - 1) * qnx) - n_neighbors) / \
            (X.shape[0] - 1 - n_neighbors)
        rnx_graph.append(rnx)
    with open("../" + folder + "/results_RNX.pkl", "wb") as fp:
        pickle.dump(rnx_graph, fp)


def path_rnx_distance_sorted(LD_data, HD_data, LD_paths, HD_paths, folder):
    results = {}
    for index in range(len(LD_data)):
        results[index] = {}
        for i in range(len(LD_paths[index])):
            LD_path = [x for x in LD_paths[index][i] if x != -1][1:]
            HD_path = [x for x in HD_paths[index][i] if x != -1][1:]
            distance = levenshteinDistanceDP(LD_path, HD_path)
            path_len = len(LD_path)

            if path_len not in results[index]:
                results[index][path_len] = [distance]
            else:
                results[index][path_len].append(distance)

        results[index] = {k: np.mean(v)
                          for k, v in sorted(results[index].items())}

    with open("../" + folder + "/results_color.pkl", "wb") as fp:
        pickle.dump(results, fp)


class DataCache:
    _instance = None
    data = {}
    with open("current_folder.txt", "r") as f:
        current_folder = f.read()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DataCache, cls).__new__(cls)
        return cls._instance

    def set_folder(self, folder):
        cls = DataCache
        cls.current_folder = folder
        # delete text in the current_folder.txt file and write the new folder name
        with open("current_folder.txt", "w") as f:
            f.write(folder)

    def load_data(self):
        print("load", self.current_folder)
        if self.current_folder is None:
            raise ValueError(
                "Folder not set. Please set a folder using `set_folder()`.")

        if self.current_folder not in self.data:
            LD_data = np.load(f"../{self.current_folder}/LD_data.npy")
            HD_data = np.load(f"../{self.current_folder}/HD_data.npy")
            LD_paths = np.load(f"../{self.current_folder}/LD_all_paths_2.npy")
            HD_paths = np.load(f"../{self.current_folder}/HD_all_paths_2.npy")
            graph = np.load(f"../{self.current_folder}/graph.npy")
            if not os.path.exists(f"../{self.current_folder}/results_color.pkl"):
                print("Calculating missing results...")
                path_rnx_distance_sorted(
                    LD_data, HD_data, LD_paths, HD_paths, self.current_folder)
            if not os.path.exists(f"../{self.current_folder}/results_RNX.pkl"):
                steps = range(30, len(LD_data), 30)
                compute_rnx(LD_data, HD_data, steps, self.current_folder)
            with open(f"../{self.current_folder}/results_color.pkl", "rb") as fp:
                results = pickle.load(fp)
            with open(f"../{self.current_folder}/results_RNX.pkl", "rb") as fp:
                rnx = pickle.load(fp)

            # Assuming rescale function is defined elsewhere
            scatter, rescaled, LD_data, means = rescale(
                LD_data, HD_data, LD_paths, HD_paths, results)

            self.data[self.current_folder] = {
                "LD_data": LD_data,
                "HD_data": HD_data,
                "LD_paths": LD_paths,
                "HD_paths": HD_paths,
                "graph": graph,
                "scatter": scatter,
                "rescaled": rescaled,
                "means": means,
                "results": results,
                "rnx": rnx,
                "folder": self.current_folder
            }
            print(f"Data loaded for {self.current_folder}.")
        return self.data[self.current_folder]


def rescale(LD_data, HD_data, LD_paths, HD_paths, results):
    longest_path_len = 0
    ys = np.zeros([len(LD_data), 500])

    for index, data in results.items():
        longest_path_len = max(longest_path_len, max(data.keys()))

    means = np.zeros(longest_path_len)
    colors = np.zeros(len(LD_data))

    for index in range(len(LD_data)):
        data = results[index]
        data = {k: v for k, v in sorted(
            data.items(), key=lambda item: item[0])}
        keys = list(data.keys())
        for i in range(len(keys)-1, -1, -1):
            data[keys[i]+2] = data[keys[i]]
        del data[1]
        del data[0]
        data = {k: v for k, v in sorted(
            data.items(), key=lambda item: item[0])}
        y = np.array(list(data.values())) / list(data.keys()) - \
            ((np.array(list(data.keys())) - 1.7) / (np.array(list(data.keys()))))

        ys[index] = np.pad(-y, (0, 500 - len(y)), "constant")

    ys_transpose = ys[:index].T
    means = np.zeros(longest_path_len)

    for i in range(longest_path_len):
        tempo = np.trim_zeros(np.sort(ys_transpose[i]))
        means[i] = np.mean(tempo)

    for i in range(index+1):
        colors[i] = np.mean(ys[i][:len(means)] - means)

    scatter_colored_points = go.Scatter(
        x=LD_data[:, 0],
        y=LD_data[:, 1],
        mode="markers",
        marker=dict(
            size=5,
            color=colors,
            colorscale="Viridis",
            showscale=True,
        ),
        name="Colored Points",
    )

    rescaled_line_plot = go.Scatter(
        x=np.arange(2, len(means) + 2),
        y=means,
        mode="lines",
        line=dict(color="blue"),
        name="Rescaled Means",
    )

    return scatter_colored_points, rescaled_line_plot, LD_data, means
