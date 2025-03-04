import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors
from Utils import *

folder = 't-SNE_COIL-20_data'
# load data
LD_data = np.load('./'+folder+'/LD_data.npy')
HD_data = np.load('./'+folder+'/HD_data.npy')
# load paths
HD_paths = np.load('./'+folder+'/HD_all_paths.npy')
HD_paths_2 = np.load('./'+folder+'/HD_all_paths_2.npy')
LD_paths = np.load('./'+folder+'/LD_all_paths.npy')
LD_paths_2 = np.load('./'+folder+'/LD_all_paths_2.npy')
# load distance matrix
LD_distance_matrix = np.load('./'+folder+'/LD_distance_matrix.npy')
HD_distance_matrix = np.load('./'+folder+'/HD_distance_matrix.npy')


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


# for folder in ['t-SNE_COIL-20_data', 'PCA_COIL-20_data', 'UMAP_COIL-20_data', 'Isomap_COIL-20_data', 'MDS_COIL-20_data']:
#     LD_data = np.load('./'+folder+'/LD_data.npy')
#     HD_data = np.load('./'+folder+'/HD_data.npy')
#     rnx = []
#     for i in range(1, len(LD_data), 30):
#         print(i)
#         rnx.append(compute_rnx(HD_data, LD_data, i))
#     plt.plot(range(1, len(LD_data), 30), rnx)
# plt.xscale('log')
# plt.show()
# load graph
# print(edges)
# for edge in edges:
#     plt.plot([X_reduced[edge[0]][0], X_reduced[edge[1]][0]], [
#              X_reduced[edge[0]][1], X_reduced[edge[1]][1]], c='black', linewidth=0.5)
# edges = delaunay_graph(LD_data)
# print(edges)
# display_delunay_graph(edges, LD_data)


def recompute(LD_data):

    X_reduced = LD_data
    X_reduced_reshaped = (X_reduced - np.min(X_reduced)) / \
        (np.max(X_reduced) - np.min(X_reduced)) * 100
    _, _, edges = alpha_shape(X_reduced_reshaped, alpha=0.00001)

    distance_matrix_LD = compute_distance_matrix(X_reduced)
    graph = np.zeros((len(X_reduced), len(X_reduced)))

    for i in range(len(graph)):
        graph[i][i] = 0  # remove self loop
    for edge in edges:
        graph[edge[0]][edge[1]] = distance_matrix_LD[edge[0]][edge[1]]
        graph[edge[1]][edge[0]] = distance_matrix_LD[edge[1]][edge[0]]

    index = 0

    plt.scatter(LD_data[:, 0], LD_data[:, 1], s=2, c='black')
    plt.scatter(LD_data[:, 0][index], LD_data[:, 1][index], s=10, c='r')

    paths = dijkstra(graph, index)

    for i in range(len(paths)):
        path = paths[i]
        # remove all -1
        path = [x for x in path if x != -1]
        for j in range(len(path)-1):
            plt.plot([LD_data[path[j]][0], LD_data[path[j+1]][0]],
                     [LD_data[path[j]][1], LD_data[path[j+1]][1]], c='b', linewidth=0.5)
    plt.show()


def path_rnx_path_length_sorted(LD_data, HD_data, LD_paths, HD_paths):
    results = {}
    for index in range(len(LD_data)):
        for i in range(index, len(LD_paths[index])):
            LD_path = LD_paths[index][i]
            HD_path = HD_paths[index][i]
            # remove all -1
            LD_path = [x for x in LD_path if x != -1]
            HD_path = [x for x in HD_path if x != -1]
            # remove first element
            LD_path = LD_path[1:]
            HD_path = HD_path[1:]

            for j in range(1, len(LD_path)):
                # distance = intersection of the jth elements of the two paths
                distance = len(set(LD_path[:j]).intersection(set(HD_path[:j])))
                if j not in results:
                    results[j] = [distance/j]
                else:
                    results[j].append(distance/j)

    results = dict(sorted(results.items()))
    for key in results:
        results[key] = np.mean(list(results[key]))
    print(results)
    plt.plot(results.keys(), results.values())
    plt.show()
    for key in results:
        tempo = results[key]

        maximum = max(list(results.keys()))
        results[key] = ((maximum + 1) * tempo - key) / \
            (maximum + 1 - key)
    print(results)
    plt.plot(results.keys(), results.values())
    plt.show()
    return results


# path_rnx_path_length_sorted(LD_data, HD_data, LD_paths_2, HD_paths_2)


def path_rnx_distance_sorted(LD_data, HD_data, LD_paths, HD_paths):
    results = {}
    for index in range(len(LD_data)):
        print(index)
        for i in range(index, len(LD_paths[index])):
            LD_path = LD_paths[index][i]
            HD_path = HD_paths[index][i]
            # remove all -1
            LD_path = [x for x in LD_path if x != -1]
            HD_path = [x for x in HD_path if x != -1]
            # remove first element
            LD_path = LD_path[1:]
            HD_path = HD_path[1:]

            # distance = intersection of the jth elements of the two paths
            distance = levenshteinDistanceDP(LD_path, HD_path)
            true_distance = np.linalg.norm(
                LD_data[index] - LD_data[i])
            if true_distance not in results:
                results[true_distance] = [distance]
            else:
                results[true_distance].append(distance)

    results = dict(sorted(results.items()))
    max_X = max(LD_data[:, 0])
    min_X = min(LD_data[:, 0])
    max_Y = max(LD_data[:, 1])
    min_Y = min(LD_data[:, 1])
    max_distance = np.linalg.norm([max_X-min_X, max_Y-min_Y])

    # make 100 buckets
    buckets = range(0, 100)
    fianl_results = {}
    results = {key: np.mean(results[key]) for key in results}

    for key in results:
        bucket_key = int(key/max_distance*100)
        if bucket_key not in fianl_results:
            fianl_results[bucket_key] = [results[key]]
        else:
            fianl_results[bucket_key].append(results[key])

    for key in fianl_results:
        fianl_results[key] = np.mean(fianl_results[key])
    print(fianl_results)
    plt.plot(fianl_results.keys(), fianl_results.values())
    return fianl_results


# for folder in ['t-SNE_COIL-20_data', 'PCA_COIL-20_data', 'UMAP_COIL-20_data', 'Isomap_COIL-20_data', 'MDS_COIL-20_data']:
#     LD_data = np.load('./'+folder+'/LD_data.npy')
#     HD_data = np.load('./'+folder+'/HD_data.npy')
#     LD_paths_2 = np.load('./'+folder+'/LD_all_paths_2.npy')
#     HD_paths_2 = np.load('./'+folder+'/HD_all_paths_2.npy')
#     path_rnx_distance_sorted(LD_data, HD_data, LD_paths_2, HD_paths_2)
#
# plt.legend(['t-SNE', 'PCA', 'UMAP', 'Isomap', 'MDS'])
# plt.show()


def path_edit_distance(LD_data, HD_data, LD_paths, HD_paths):
    results = {}
    for index in range(len(LD_data)):
        for i in range(index, len(LD_paths[index])):
            LD_path = LD_paths[index][i]
            HD_path = HD_paths[index][i]
            # remove all -1
            LD_path = [x for x in LD_path if x != -1]
            HD_path = [x for x in HD_path if x != -1]
            # remove first element
            LD_path = LD_path[1:]
            HD_path = HD_path[1:]

            # distance = intersection of the jth elements of the two paths
            distance = levenshteinDistanceDP(LD_path, HD_path)
            if len(LD_path) not in results:
                results[len(LD_path)] = [distance]
            else:
                results[len(LD_path)].append(distance)

    results = dict(sorted(results.items()))
    results = {key: np.mean(results[key]) for key in results}
    plt.plot(results.keys(), results.values())
    print(results)


# for folder in ['t-SNE_COIL-20_data', 'PCA_COIL-20_data', 'UMAP_COIL-20_data', 'Isomap_COIL-20_data', 'MDS_COIL-20_data']:
for folder in ['t-SNE_MNIST_data', 'PCA_MNIST_data', 'UMAP_MNIST_data', 'Isomap_MNIST_data', 'MDS_MNIST_data']:
    print(f"loading {folder}")
    LD_data = np.load('./'+folder+'/LD_data.npy')
    HD_data = np.load('./'+folder+'/HD_data.npy')
    LD_paths_2 = np.load('./'+folder+'/LD_all_paths_2.npy')
    HD_paths_2 = np.load('./'+folder+'/HD_all_paths_2.npy')
    path_edit_distance(LD_data, HD_data, LD_paths_2, HD_paths_2)
plt.legend(['t-SNE', 'PCA', 'UMAP', 'Isomap', 'MDS'])
plt.show()


def plot_path_from_index_colors(LD_data, LD_paths, index):
    cmap = plt.get_cmap('viridis')

    coutour = []
    max_len = 0
    for i in range(len(LD_paths)):

        LD_path = LD_paths[index][i]
        LD_path = [x for x in LD_path if x != -1]
        max_len = max(max_len, len(LD_path))
    norm = plt.Normalize(0, max_len)
    for i in range(len(LD_paths[index])):
        LD_path = LD_paths[index][i]
        # remove all -1
        LD_path = [x for x in LD_path if x != -1]
        # cast to int array
        LD_path = [int(x) for x in LD_path]
        if len(LD_path) != 0:
            plt.scatter(LD_data[LD_path[-1], 0],
                        LD_data[LD_path[-1], 1], c=cmap(norm(len(LD_path))), s=5)
            # add a number crecponing the path length on the points
            if len(LD_path) == 20:
                coutour.append(LD_path[-1])

                # plt.text(LD_data[LD_path[-1], 0],
                #          LD_data[LD_path[-1], 1], str(len(LD_path)), color='r')

    # sort the coutour by angle value with the center point
    coutour = sorted(coutour, key=lambda x: np.arctan2(
        LD_data[x, 1]-LD_data[index, 1], LD_data[x, 0]-LD_data[index, 0]))
    print(coutour)
    # display the coutour
    for i in range(len(coutour)-1):
        plt.plot([LD_data[coutour[i], 0], LD_data[coutour[i+1], 0]], [
            LD_data[coutour[i], 1], LD_data[coutour[i+1], 1]], c='r', linewidth=0.5)

    # compute nearest neighbors of the index point

    neighbors = np.argsort(LD_distance_matrix[index])

    # plot a circle around the index point with the radius of the distance to the 10th nearest neighbor
    print(LD_distance_matrix[index][neighbors[500]])
    circle = plt.Circle(LD_data[index], LD_distance_matrix[index]
                        [neighbors[500]], fill=False, color='b')
    plt.gca().add_artist(circle)

    plt.scatter(LD_data[:, 0][index], LD_data[:, 1][index], s=20, c='r')
    # display color legend
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=plt.gca(), label='Path index')

    plt.show()


recompute(LD_data)
plot_path_from_index_colors(LD_data, LD_paths_2, 0)
