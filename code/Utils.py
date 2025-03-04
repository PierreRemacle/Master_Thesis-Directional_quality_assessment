from numpy.random import seed
import shapely.geometry as geometry
import math
from shapely.ops import cascaded_union, polygonize
from sknetwork.clustering import Louvain, get_modularity
from sklearn.cluster import AffinityPropagation
import time
from heapdict import heapdict
from alive_progress import alive_bar
import heapq
# from numba import jit
from astar import AStar
import os
import Levenshtein as lev
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import warnings
from scipy.spatial import Delaunay
from sklearn import datasets

from sklearn.manifold import TSNE, Isomap, MDS
warnings.filterwarnings("ignore")

HD_DATA_NAME = ""
LD_DATA_NAME = ""

with open("../.env", "r") as f:
    for line in f:
        key, value = line.split("=")
        if key == "HD_DATA_NAME":
            HD_DATA_NAME = value.strip()
        if key == "LD_DATA_NAME":
            LD_DATA_NAME = value.strip()

HD_data = []
LD_data = []
# HD_data = pd.read_csv('../DATA/' + HD_DATA_NAME + '.csv')
# HD_data = HD_data.drop(HD_data.columns[0], axis=1)
# HD_data = HD_data.to_numpy()
#
# LD_data = pd.read_csv('../DATA/' + LD_DATA_NAME + '.csv').to_numpy()[:, [1, 2]]

# mnist = datasets.fetch_openml('mnist_784')
# # only 3000 samples are used for performance reasons
# mnist.data = mnist.data[:3000]
# mnist.target = mnist.target[:3000]
# HD_data = mnist.data / 255.0  # Normalize
# HD_data = np.array(HD_data, dtype=float)
# LD_data = TSNE(n_components=2, random_state=42).fit_transform(HD_data)


class BasicAStar(AStar):
    def __init__(self, matrix, distance_matrix, LD_data):
        self.adjmatrix = matrix
        self.distance_matrix = distance_matrix
        self.LD_data = LD_data

    def neighbors(self, n):
        return [x for x, v in enumerate(self.adjmatrix[n]) if v > 0]

    def distance_between(self, n1, n2):
        if n1 > n2:
            n1, n2 = n2, n1
        return self.adjmatrix[n1][n2]

    def heuristic_cost_estimate(self, current, goal):
        return self.distance_matrix[current][goal]

    def is_goal_reached(self, current, goal):
        return current == goal


def compute_distance_matrix(data):
    distance_matrix = np.zeros((len(data), len(data)))
    with alive_bar(int((len(data)*(len(data)-1))/2)) as bar:
        for i in range(len(data)):
            for j in range(i+1, len(data)):
                distance_matrix[i][j] = np.linalg.norm(data[i] - data[j])
                distance_matrix[j][i] = distance_matrix[i][j]
                bar()
    return distance_matrix


def delaunay_graph(LD_data=LD_data):
    tri = Delaunay(LD_data)
    grah = np.zeros((len(LD_data), len(LD_data)))
    for i in range(len(tri.simplices)):
        for j in range(3):
            grah[tri.simplices[i][j]][tri.simplices[i]
                                      [(j+1) % 3]] = np.linalg.norm(LD_data[tri.simplices[i][j]] - LD_data[tri.simplices[i][(j+1) % 3]])**2
            grah[tri.simplices[i][(j+1) % 3]][tri.simplices[i][j]
                                              ] = grah[tri.simplices[i][j]][tri.simplices[i][(j+1) % 3]]
    return grah


def display_delunay_graph(graph, LD_data=LD_data):
    fig = plt.figure()
    ax = plt.axes()
    for i in range(len(graph)):
        for j in range(i+1, len(graph[i])):
            if graph[i][j] > 0:
                ax.plot([LD_data[i][0], LD_data[j][0]], [
                        LD_data[i][1], LD_data[j][1]], color='black')
    plt.show()


def dijkstra_slow(graph, start):
    n = len(graph)
    unvisited = set(range(n))
    distance = np.full(n, np.inf)
    distance[start] = 0
    path = - np.ones((n, n), dtype=np.int64)
    path[start][0] = start
    while unvisited:
        current = min(unvisited, key=lambda node: distance[node])
        unvisited.remove(current)
        for neighbor, weight in enumerate(graph[current]):
            if weight > 0:
                new_distance = distance[current] + weight
                if new_distance < distance[neighbor]:
                    distance[neighbor] = new_distance
                    indice = 0
                    while path[current][indice] != -1:
                        path[neighbor][indice] = path[current][indice]
                        indice += 1
                    path[neighbor][indice] = neighbor
    return path


def dijkstra(graph, start):
    n = len(graph)
    distance = np.full(n, np.inf)
    distance[start] = 0
    path = -np.ones((n, n), dtype=np.int64)
    path[start][0] = start

    priority_queue = [(0, start)]  # (distance, node)
    while priority_queue:
        dist_to_current, current = heapq.heappop(priority_queue)
        if dist_to_current > distance[current]:
            continue
        for neighbor, weight in enumerate(graph[current]):
            if weight > 0:
                new_distance = dist_to_current + weight
                if new_distance < distance[neighbor]:
                    distance[neighbor] = new_distance
                    heapq.heappush(priority_queue, (new_distance, neighbor))
                    path[neighbor][:] = path[current][:]
                    path[neighbor, np.count_nonzero(
                        path[neighbor] != -1)] = neighbor
    return path


def dijkstrafib(graph, start):
    n = len(graph)
    distance = np.full(n, np.inf)
    distance[start] = 0
    path = -np.ones((n, n), dtype=np.int64)
    path[start][0] = start

    priority_queue = heapdict({start: 0})  # {node: distance}
    while priority_queue:
        current, dist_to_current = priority_queue.popitem()
        for neighbor, weight in enumerate(graph[current]):
            if weight > 0:
                new_distance = dist_to_current + weight
                if new_distance < distance[neighbor]:
                    distance[neighbor] = new_distance
                    priority_queue[neighbor] = new_distance
                    path[neighbor][:] = path[current][:]
                    path[neighbor, np.count_nonzero(
                        path[neighbor] != -1)] = neighbor
    return path


def create_HD_path(LD_path, distance_matrix, start, end):
    # best way to go through all the points
    LD_path = np.asarray(LD_path)
    LD_path = LD_path[LD_path != -1]
    distance_to_origin = np.zeros((len(LD_path), 2))
    for i in range(len(LD_path)):
        distance_to_origin[i][0] = LD_path[i]
        distance_to_origin[i][1] = distance_matrix[LD_path[i]][start]
    # for i in range(len(LD_path)):
    #     distance_to_origin[i][0] = LD_path[i]
    #     distance_to_origin[i][1] = np.linalg.norm(
    #         HD_data[LD_path[i]] - HD_data[start])

    distance_to_origin = distance_to_origin[distance_to_origin[:, 1].argsort()]
    HD_path = [distance_to_origin[i][0]
               for i in range(len(LD_path))]
    HD_path = np.asarray(HD_path, dtype=int)
    # print(distance_to_origin)
    # print(LD_path)
    # print(HD_path)
    # print("------------------")
    return HD_path


def create_HD_path2(LD_path, distance_matrix, start, end):

    start = LD_path[0]
    HD_path = - np.ones(len(LD_path), dtype=int)
    HD_path[0] = start
    path_here = np.asarray(LD_path[1:])
    index = 1  # index of the HD_path
    while len(path_here) > 0:
        HD_neigh = distance_matrix[start][path_here]
        index_of_min = np.argmin(HD_neigh)
        closest = path_here[index_of_min]
        HD_path[index] = closest
        path_here = path_here[path_here != closest]
        start = closest
        # if start == end:
        #     break
        index += 1
    return HD_path[HD_path != -1]


def levenshteinDistanceDP(token1, token2, printMatrix=False):
    if printMatrix:

        distances = np.zeros((len(token1) + 1, len(token2) + 1))

        for t1 in range(len(token1) + 1):
            distances[t1][0] = t1

        for t2 in range(len(token2) + 1):
            distances[0][t2] = t2

        a = 0
        b = 0
        c = 0

        for t1 in range(1, len(token1) + 1):
            for t2 in range(1, len(token2) + 1):
                if (token1[t1-1] == token2[t2-1]):
                    distances[t1][t2] = distances[t1 - 1][t2 - 1]
                else:
                    a = distances[t1][t2 - 1]
                    b = distances[t1 - 1][t2]
                    c = distances[t1 - 1][t2 - 1]

                    if (a <= b and a <= c):
                        distances[t1][t2] = a + 1
                    elif (b <= a and b <= c):
                        distances[t1][t2] = b + 1
                    else:
                        distances[t1][t2] = c + 1

        printDistances(distances, len(token1), len(token2), token1, token2)
        return distances[len(token1)][len(token2)]
    else:

        return lev.distance(token1, token2)


def printDistances(distances, len1, len2, token1, token2):
    print("   ", end=" ")
    token1 = [" "] + token1
    for t1 in range(len(token2)):
        print(token2[t1], end=" ")
    print("\n")
    for t1 in range(len1 + 1):
        print(token1[t1], end=" ")
        for t2 in range(len2 + 1):
            print(int(distances[t1][t2]), end=" ")
        print("\n")


def random_walk(knn, graph, nbr_itter=10, nbr_walk=100):
    results = {}
    walks = []
    for i in range(nbr_walk):
        start = np.random.randint(0, len(LD_data))
        previous_node = start
        LD_path = [start]
        for j in range(nbr_itter):
            next_node = np.random.choice(graph[previous_node].nonzero()[0])
            LD_path.append(next_node)
            HD_path = create_HD_path(LD_path, start, next_node)
            penality = levenshteinDistanceDP(LD_path, HD_path)
            if results.get(len(LD_path)) is None:
                results[len(LD_path)] = [penality]
            else:
                results[len(LD_path)].append(penality)
            previous_node = next_node
        walks.append((LD_path, penality))
    for i in results:
        results[i] = sum(results[i]) / len(results[i])
    return results, walks


def compute_cost(HD_data, path):
    cost = 0
    for i in range(len(path) - 1):
        cost += np.linalg.norm(HD_data.iloc[path[i]] -
                               HD_data.iloc[path[i + 1]])
    return cost


def LDHDPath(graph, start, end, distance_matrix_LD, distance_matrix_HD):
    LD_path = list(BasicAStar(graph, distance_matrix_LD,
                   LD_data).astar(start, end))
    HD_path = create_HD_path(LD_path, distance_matrix_HD, start, end)
    return LD_path, HD_path


def LDHDPathAll(graph, distance_matrix, start=0, LD_data=LD_data, HD_data=HD_data):
    LD_paths = dijkstra(graph, start)
    HD_paths = - np.ones((len(LD_data), 200), dtype=int)

    for end in range(len(LD_data)):
        if start != end:
            LD_path = LD_paths[end]
            HD_path = create_HD_path2(LD_path, distance_matrix, start, end)
            for i in range(len(HD_path)):
                HD_paths[end][i] = HD_path[i]
        else:
            HD_paths[end][0] = start
    return LD_paths, HD_paths


def HDHDpathAll(graph, distance_matrix, start=0, LD_data=LD_data, HD_data=HD_data):
    LD_paths = dijkstra(graph, start)
    HD_paths_forwards = - np.ones((len(LD_data), len(LD_data)), dtype=int)
    HD_paths_backwards = - np.ones((len(LD_data), len(LD_data)), dtype=int)
    for end in range(len(LD_data)):
        if start != end:
            LD_path = LD_paths[end]
            HD_path = create_HD_path(LD_path, distance_matrix, start, end)
            for i in range(len(HD_path)):
                HD_paths_forwards[end][i] = HD_path[i]
            HD_path = create_HD_path(
                LD_path[::-1], distance_matrix, end, start)
            for i in range(len(HD_path)):
                HD_paths_backwards[end][i] = HD_path[i]
    return HD_paths_forwards, HD_paths_backwards


def LDHDPathcompare(graph, start=0, end=1, LD_data=LD_data, HD_data=HD_data):
    LD_path = list(BasicAStar(graph, LD_data.to_numpy()
                   [:, [1, 2]]).astar(start, end))
    LD_path_d = dijkstra(graph, start, end)
    if LD_path != LD_path_d:
        print("Error")
    HD_path = create_HD_path(LD_path, start, end)
    HD_path_2 = create_HD_path(LD_path_d, start, end)
    return LD_path, HD_path, LD_path_d, HD_path_2


def LDHDCompare(LD_path, HD_path):

    # find longest similar subsequence
    similar_sequence = []
    size = len(LD_path)

    pointer1 = 0
    pointer2 = 0

    while pointer1 < size:
        while LD_path[pointer1] == HD_path[pointer2]:
            pointer2 += 1
        similar_sequence.append(LD_path[pointer1: pointer2])
        pointer1 = pointer2
        pointer2 += 1
    return similar_sequence


def ALL_path(HD_data=HD_data, LD_data=LD_data, n_neighbors=30):
    graph = delaunay_graph(LD_data)
    distance_matrix = compute_distance_matrix(HD_data)
    results = {}
    localisation_of_errors = np.zeros((len(HD_data), len(LD_data)))
    frequence_of_errors = np.ones((len(HD_data), len(LD_data)))
    with alive_bar(int((len(HD_data) * (len(HD_data)-1)/2))) as bar:
        for i in range(len(HD_data)):
            for j in range(i+1, len(LD_data)):
                if i != j:
                    LD_path, HD_path = LDHDPath(graph, distance_matrix, i, j)
                    for k in range(len(LD_path)-1):
                        first = min(LD_path[k], LD_path[k+1])
                        second = max(LD_path[k], LD_path[k+1])
                        if LD_path[k+1] != HD_path[k+1]:
                            localisation_of_errors[first][second] += 1
                        frequence_of_errors[first][second] += 1
                    # compute lavenshtein distance
                    if results.get(len(LD_path)) is None:
                        results[len(LD_path)] = [
                            levenshteinDistanceDP(LD_path, HD_path)]
                    else:
                        results[len(LD_path)].append(
                            levenshteinDistanceDP(LD_path, HD_path))
                bar()
    return results, localisation_of_errors/frequence_of_errors


def ALL_path_2(HD_data, LD_data, graph=None, save_folder='results_folder'):
    # Create folder if it doesn't exist
    os.makedirs(save_folder, exist_ok=True)

    # Initialize results
    results = {}
    distance_matrix = compute_distance_matrix(HD_data)
    error_per_start = np.zeros(len(LD_data))
    localisation_of_errors = np.zeros((len(HD_data), len(LD_data)))
    frequence_of_errors = np.ones((len(HD_data), len(LD_data)))

    # Progress bar
    with alive_bar(len(HD_data)) as bar:
        for i in range(len(HD_data)):
            LD_paths, HD_paths = LDHDPathAll(
                graph, distance_matrix, i, LD_data, HD_data)
            for j in range(i + 1, len(LD_paths)):
                LD_path = LD_paths[j][LD_paths[j] != -1]
                HD_path = HD_paths[j][HD_paths[j] != -1]

                for k in range(len(LD_path) - 1):
                    first, second = min(
                        LD_path[k], LD_path[k + 1]), max(LD_path[k], LD_path[k + 1])
                    if LD_path[k + 1] != HD_path[k + 1]:
                        localisation_of_errors[first][second] += 1
                    frequence_of_errors[first][second] += 1

                # Calculate intersection-based distance
                intersection_distance = len(
                    set(LD_path).intersection(set(HD_path))) / len(LD_path)

                # Calculate Levenshtein distance
                levenshtein_distance = levenshteinDistanceDP(LD_path, HD_path)

                # Store both distances in the results dictionary
                if len(LD_path) not in results:
                    results[len(LD_path)] = [
                        (levenshtein_distance, intersection_distance)]
                else:
                    results[len(LD_path)].append(
                        (levenshtein_distance, intersection_distance))

                # Accumulate the intersection-based error
                error_per_start[i] += intersection_distance

            bar()

    # Calculate overall quality based on intersection distance
    overall_quality = sum(error_per_start) / len(LD_data)
    print("Overall quality (intersection-based):", overall_quality)

    # Save components to the specified folder
    np.save(os.path.join(save_folder, 'HD_data.npy'), HD_data)
    np.save(os.path.join(save_folder, 'LD_data.npy'), LD_data)
    np.save(os.path.join(save_folder, 'graph.npy'), graph)
    np.save(os.path.join(save_folder, 'distance_matrix.npy'), distance_matrix)
    np.save(os.path.join(save_folder, 'localisation_of_errors.npy'),
            localisation_of_errors)
    np.save(os.path.join(save_folder, 'frequence_of_errors.npy'),
            frequence_of_errors)
    np.save(os.path.join(save_folder, 'error_per_start.npy'), error_per_start)

    # Save results dictionary containing both distances
    np.save(os.path.join(save_folder, 'results.npy'), results)

    # Optionally, save any plots (example of error localization heatmap)
    plt.imshow(localisation_of_errors / frequence_of_errors,
               cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title('Error Localization Heatmap')
    plt.savefig(os.path.join(save_folder, 'error_heatmap.png'))
    plt.close()

    return results, localisation_of_errors / frequence_of_errors, error_per_start


def ALL_path_3(HD_data, LD_data, graph=None, save_folder='results_folder'):
    # Create folder if it doesn't exist
    save_folder = save_folder + "_data"
    os.makedirs(save_folder, exist_ok=True)

    # Initialize results
    distance_matrix = compute_distance_matrix(HD_data)
    LD_distance_matrix = compute_distance_matrix(LD_data)
    edges = graph
    graph = np.zeros((len(LD_data), len(LD_data)))
    # graph = graph**2 + np.max(graph) * 100
    for i in range(len(graph)):
        graph[i][i] = 0  # remove self loop
    for edge in edges:
        graph[edge[0]][edge[1]] = LD_distance_matrix[edge[0]][edge[1]]
        graph[edge[1]][edge[0]] = LD_distance_matrix[edge[1]][edge[0]]
    error_per_start = np.zeros(len(LD_data))
    localisation_of_errors = np.zeros((len(HD_data), len(LD_data)))
    frequence_of_errors = np.ones((len(HD_data), len(LD_data)))
    LD_all_paths = -np.ones((len(LD_data), len(LD_data), 200))
    LD_all_paths_2 = -np.ones((len(LD_data), len(LD_data), 200))
    HD_all_paths = -np.ones((len(LD_data), len(LD_data), 200))
    HD_all_paths_2 = -np.ones((len(LD_data), len(LD_data), 200))
    # Progress bar
    with alive_bar(len(HD_data)) as bar:
        for i in range(len(HD_data)):
            LD_paths, HD_paths = LDHDPathAll(
                graph, distance_matrix, i, LD_data, HD_data)
            LD_paths_2 = []
            for j in range(i + 1, len(LD_paths)):
                LD_path_2 = create_HD_path(
                    LD_paths[j], LD_distance_matrix, i, j)
                HD_path_2 = create_HD_path(
                    LD_paths[j], distance_matrix, i, j)
                # reshape by adding -1 to get same lengths as LD_path
                LD_path_2 = np.append(
                    LD_path_2, [-1] * (200 - len(LD_path_2)))
                HD_path_2 = np.append(
                    HD_path_2, [-1] * (200 - len(HD_path_2)))
                LD_all_paths_2[i][j] = LD_path_2
                LD_all_paths_2[j][i] = LD_path_2
                HD_all_paths_2[i][j] = HD_path_2
                HD_all_paths_2[j][i] = HD_path_2
                LD_all_paths[i][j] = LD_paths[j][:200]
                LD_all_paths[j][i] = LD_paths[j][:200]
                HD_all_paths[i][j] = HD_paths[j][:200]
                HD_all_paths[j][i] = HD_paths[j][:200]

            # LD_all_paths.append(LD_paths)
            # HD_all_paths.append(HD_paths)

            bar()

    # Calculate overall quality based on intersection distance
    overall_quality = sum(error_per_start) / len(LD_data)
    print("Overall quality (intersection-based):", overall_quality)
    # save all the paths for further analysis
    # print(np.array(LD_all_paths).shape)
    np.save(os.path.join(save_folder, 'LD_all_paths.npy'), LD_all_paths)
    # print(LD_all_paths_2.shape)
    np.save(os.path.join(save_folder, 'LD_all_paths_2.npy'), LD_all_paths_2)
    # print(np.array(HD_all_paths).shape)
    np.save(os.path.join(save_folder, 'HD_all_paths.npy'), HD_all_paths)
    # print(HD_all_paths_2.shape)
    np.save(os.path.join(save_folder, 'HD_all_paths_2.npy'), HD_all_paths_2)
    # Save components to the specified folder
    np.save(os.path.join(save_folder, 'HD_data.npy'), HD_data)
    np.save(os.path.join(save_folder, 'LD_data.npy'), LD_data)
    np.save(os.path.join(save_folder, 'graph.npy'), graph)
    np.save(os.path.join(save_folder, 'HD_distance_matrix.npy'), distance_matrix)
    np.save(os.path.join(save_folder, 'LD_distance_matrix.npy'), LD_distance_matrix)

    return 0


def ALL_path_3_convex_hull(HD_data, LD_data, edge_points, graph=None, save_folder='results_folder'):
    # Create folder if it doesn't exist
    save_folder = save_folder + "_data"
    os.makedirs(save_folder, exist_ok=True)

    # Initialize results
    distance_matrix = compute_distance_matrix(HD_data)
    LD_distance_matrix = compute_distance_matrix(LD_data)
    edges = graph
    graph = np.zeros((len(LD_data), len(LD_data)))
    # graph = graph**2 + np.max(graph) * 100
    for i in range(len(graph)):
        graph[i][i] = 0  # remove self loop
    for edge in edges:
        graph[edge[0]][edge[1]] = LD_distance_matrix[edge[0]][edge[1]]
        graph[edge[1]][edge[0]] = LD_distance_matrix[edge[1]][edge[0]]
    error_per_start = np.zeros(len(LD_data))
    localisation_of_errors = np.zeros((len(HD_data), len(LD_data)))
    frequence_of_errors = np.ones((len(HD_data), len(LD_data)))
    LD_all_paths = -np.ones((len(edge_points), len(LD_data), 200))
    LD_all_paths_2 = -np.ones((len(edge_points), len(LD_data), 200))
    HD_all_paths = -np.ones((len(edge_points), len(LD_data), 200))
    HD_all_paths_2 = -np.ones((len(edge_points), len(LD_data), 200))
    # Progress bar
    with alive_bar(len(HD_data)) as bar:
        for i in range(len(edge_points)):
            LD_paths, HD_paths = LDHDPathAll(
                graph, distance_matrix, start[i], LD_data, HD_data)
            LD_paths_2 = []
            for j in range(len(LD_paths)):
                LD_path_2 = create_HD_path(
                    LD_paths[j], LD_distance_matrix, start[i], j)
                HD_path_2 = create_HD_path(
                    LD_paths[j], distance_matrix, start[i], j)
                # reshape by adding -1 to get same lengths as LD_path
                LD_path_2 = np.append(
                    LD_path_2, [-1] * (200 - len(LD_path_2)))
                HD_path_2 = np.append(
                    HD_path_2, [-1] * (200 - len(HD_path_2)))
                LD_all_paths_2[i][j] = LD_path_2
                LD_all_paths_2[j][i] = LD_path_2
                HD_all_paths_2[i][j] = HD_path_2
                HD_all_paths_2[j][i] = HD_path_2
                LD_all_paths[i][j] = LD_paths[j][:200]
                LD_all_paths[j][i] = LD_paths[j][:200]
                HD_all_paths[i][j] = HD_paths[j][:200]
                HD_all_paths[j][i] = HD_paths[j][:200]

            # LD_all_paths.append(LD_paths)
            # HD_all_paths.append(HD_paths)

            bar()

    # Calculate overall quality based on intersection distance
    overall_quality = sum(error_per_start) / len(LD_data)
    print("Overall quality (intersection-based):", overall_quality)
    # save all the paths for further analysis
    # print(np.array(LD_all_paths).shape)
    np.save(os.path.join(save_folder, 'LD_all_paths.npy'), LD_all_paths)
    # print(LD_all_paths_2.shape)
    np.save(os.path.join(save_folder, 'LD_all_paths_2.npy'), LD_all_paths_2)
    # print(np.array(HD_all_paths).shape)
    np.save(os.path.join(save_folder, 'HD_all_paths.npy'), HD_all_paths)
    # print(HD_all_paths_2.shape)
    np.save(os.path.join(save_folder, 'HD_all_paths_2.npy'), HD_all_paths_2)
    # Save components to the specified folder
    np.save(os.path.join(save_folder, 'HD_data.npy'), HD_data)
    np.save(os.path.join(save_folder, 'LD_data.npy'), LD_data)
    np.save(os.path.join(save_folder, 'graph.npy'), graph)
    np.save(os.path.join(save_folder, 'HD_distance_matrix.npy'), distance_matrix)
    np.save(os.path.join(save_folder, 'LD_distance_matrix.npy'), LD_distance_matrix)

    return 0

# def ALL_path_enveloppe(HD_data=HD_data, LD_data=LD_data, graph=delaunay_graph(LD_data), enveloppe=[]):
#
#     results = {}
#     distance_matrix = compute_distance_matrix(HD_data)
#     error_per_start = np.zeros(len(LD_data))
#     localisation_of_errors = np.zeros((len(HD_data), len(LD_data)))
#     frequence_of_errors = np.ones((len(HD_data), len(LD_data)))
#     with alive_bar(int(len(enveloppe))) as bar:
#         for i in enveloppe:
#             LD_paths, HD_paths = LDHDPathAll(
#                 graph, distance_matrix, i, LD_data, HD_data)
#             for j in range(i+1, len(LD_paths)):
#                 LD_path = LD_paths[j]
#                 LD_path = LD_path[LD_path != -1]
#                 HD_path = HD_paths[j]
#                 HD_path = HD_path[HD_path != -1]
#                 for k in range(len(LD_path)-1):
#                     first = min(LD_path[k], LD_path[k+1])
#                     second = max(LD_path[k], LD_path[k+1])
#                     frequence_of_errors[first][second] += 1
#                 # compute lavenshtein distance
#                 distance = levenshteinDistanceDP(LD_path, HD_path)/len(LD_path)
#                 error_per_start[i] += distance
#                 if results.get(len(LD_path)) is None:
#                     results[len(LD_path)] = [distance]
#                 else:
#                     results[len(LD_path)].append(distance)
#             bar()
#     return results, localisation_of_errors/frequence_of_errors, error_per_start
#
#
# def ALL_path_enveloppe_HDcompare(HD_data=HD_data, LD_data=LD_data, graph=delaunay_graph(LD_data), enveloppe=[]):
#
#     results = {}
#     distance_matrix = compute_distance_matrix(HD_data)
#     error_per_start = np.zeros(len(LD_data))
#     localisation_of_errors = np.zeros((len(HD_data), len(LD_data)))
#     frequence_of_errors = np.ones((len(HD_data), len(LD_data)))
#     with alive_bar(int(len(enveloppe))) as bar:
#
#         for i in enveloppe:
#             print(i)
#             HD_paths_forwards, HD_paths_backwards = HDHDpathAll(
#                 graph, distance_matrix, i)
#             for j in range(i+1, len(HD_paths_forwards)):
#                 HD_path1 = HD_paths_forwards[j]
#                 HD_path1 = HD_path1[HD_path1 != -1]
#                 HD_path2 = HD_paths_backwards[j]
#                 HD_path2 = HD_path2[HD_path2 != -1]
#                 for k in range(len(HD_path1)-1):
#                     first = min(HD_path1[k], HD_path1[k+1])
#                     second = max(HD_path1[k], HD_path1[k+1])
#                     if HD_path1[k+1] != HD_path2[k+1]:
#                         localisation_of_errors[first][second] += 1
#                     frequence_of_errors[first][second] += 1
#                 # compute lavenshtein distance
#                 distance = levenshteinDistanceDP(HD_path1, HD_path2)
#                 error_per_start[i] += distance
#                 if results.get(len(HD_path1)) is None:
#                     results[len(HD_path1)] = [distance]
#                 else:
#                     results[len(HD_path1)].append(distance)
#             bar()
#     print("overall quality : ", sum(results)/len(results))
#     return results, localisation_of_errors/frequence_of_errors, error_per_start
#
#
# def random_selection_path_2(HD_data=HD_data, LD_data=LD_data, nbr_itter=200, graph=delaunay_graph(LD_data)):
#     results = {}
#     error_per_start = np.zeros(len(LD_data))
#     localisation_of_errors = np.zeros((len(HD_data), len(LD_data)))
#     frequence_of_errors = np.ones((len(HD_data), len(LD_data)))
#     with alive_bar(nbr_itter**2) as bar:
#         for i in range(nbr_itter):
#             start = np.random.randint(0, len(LD_data))
#             LD_paths, HD_paths = LDHDPathAll(graph, start)
#             for j in range(nbr_itter):
#                 end = np.random.randint(0, len(LD_data))
#                 if start == end:
#                     end = (end + 1) % len(LD_data)
#                 LD_path = LD_paths[end]
#                 LD_path = LD_path[LD_path != -1]
#                 HD_path = HD_paths[end]
#                 HD_path = HD_path[HD_path != -1]
#                 for k in range(len(LD_path)-1):
#                     first = min(LD_path[k], LD_path[k+1])
#                     second = max(LD_path[k], LD_path[k+1])
#                     if LD_path[k+1] != HD_path[k+1]:
#                         localisation_of_errors[first][second] += 1
#                     frequence_of_errors[first][second] += 1
#                 # compute lavenshtein distance
#                 distance = levenshteinDistanceDP(LD_path, HD_path)
#                 error_per_start[start] += distance
#                 if results.get(len(LD_path)) is None:
#                     results[len(LD_path)] = [distance]
#                 else:
#                     results[len(LD_path)].append(distance)
#                 bar()
#     return results, localisation_of_errors/frequence_of_errors, error_per_start
#


def random_selection_path(HD_data=HD_data, LD_data=LD_data, nbr_itter=200):
    graph = delaunay_graph(LD_data)
    results = {}
    localisation_of_errors = np.zeros((len(HD_data), len(LD_data)))
    frequence_of_errors = np.ones((len(HD_data), len(LD_data)))
    with alive_bar(nbr_itter**2) as bar:
        for i in range(nbr_itter):
            for j in range(nbr_itter):
                start = np.random.randint(0, len(LD_data))
                end = np.random.randint(0, len(LD_data))
                if start == end:
                    end = (end + 1) % len(LD_data)
                LD_path, HD_path = LDHDPath(graph, start, end)
                for k in range(len(LD_path)-1):
                    first = min(LD_path[k], LD_path[k+1])
                    second = max(LD_path[k], LD_path[k+1])
                    if LD_path[k+1] != HD_path[k+1]:
                        localisation_of_errors[first][second] += 1
                    frequence_of_errors[first][second] += 1
                # compute lavenshtein distance
                if results.get(len(LD_path)) is None:
                    results[len(LD_path)] = [
                        levenshteinDistanceDP(LD_path, HD_path)]
                else:
                    results[len(LD_path)].append(
                        levenshteinDistanceDP(LD_path, HD_path))
                bar()
    return results, localisation_of_errors/frequence_of_errors


def find_enveloppe(LD_data):
    enveloppe = []
    baricenter = np.mean(LD_data, axis=0)
    # add baricenter to data
    LD_data = np.vstack((LD_data, baricenter))
    graph = delaunay_graph(LD_data)
    leafs = dijkstra_leaf(graph, len(LD_data)-1)
    retour = []
    for i in range(len(leafs)):
        if leafs[i]:
            retour.append(i)
    return retour, baricenter


def dijkstra_leaf(graph, start):
    n = len(graph)
    distance = np.full(n, np.inf)
    distance[start] = 0
    leafs = [False]*n
    leafs[start] = True

    priority_queue = [(0, start)]  # (distance, node)
    while priority_queue:
        dist_to_current, current = heapq.heappop(priority_queue)
        if dist_to_current > distance[current]:
            continue
        for neighbor, weight in enumerate(graph[current]):
            if weight > 0:
                new_distance = dist_to_current + weight
                if new_distance < distance[neighbor]:
                    distance[neighbor] = new_distance
                    leafs[neighbor] = True
                    leafs[current] = False
                    heapq.heappush(priority_queue, (new_distance, neighbor))
    return leafs


def ccw(A, B, C):
    return (C[1]-A[1])*(B[0]-A[0]) > (B[1]-A[1])*(C[0]-A[0])


def graham_scan(points, already_chosen=[]):
    # Find the lowest point
    start = np.argmin(points, axis=0)[1]
    # add index to points for later use
    points = np.array([[*point, i] for i, point in enumerate(points)])
    # Sort the points by the angle they and the start point make with the x-axis
    points = sorted(points, key=lambda x: (np.arctan2(
        x[1]-points[start][1], x[0]-points[start][0])))
    stack = []
    index_stack = []
    for p in points:
        if p[2] in already_chosen:
            continue
        while len(stack) > 1 and ccw(stack[-2], stack[-1], p) <= 0:
            stack.pop()
            index_stack.pop()
        stack.append(p)
        index_stack.append(int(p[2]))
    return stack, index_stack


def multiple_enveloppe(LD_data, number_of_enveloppe=1):
    already_chosen = []
    all_enveloppe = []
    all_enveloppe_index = []
    for i in range(number_of_enveloppe):
        enveloppe, index_enveloppe = graham_scan(LD_data, already_chosen)
        already_chosen += index_enveloppe
        all_enveloppe.append(enveloppe)
        all_enveloppe_index.append(index_enveloppe)
    return all_enveloppe, all_enveloppe_index


def clustering(LD_data, adjacency, number_of_cluster=1):
    louvain = Louvain()
    print(adjacency)
    labels = louvain.fit_predict(adjacency)
    print(labels)
    clusters = []
    number_of_cluster = len(af.cluster_centers_indices_)
    for i in range(number_of_cluster):
        clusters.append(LD_data[labels_ == i])
    return clusters


def alpha_shape(points, alpha):
    """
    Compute the alpha shape (concave hull) of a set of points.

    @param points: Iterable container of points.
    @param alpha: alpha value to influence the gooeyness of the border. Smaller
                  numbers don't fall inward as much as larger numbers. Too large,
                  and you lose everything!
    """
    if len(points) < 4:
        # When you have a triangle, there is no sense in computing an alpha
        # shape.
        return geometry.MultiPoint(list(points)).convex_hull

    def add_edge(edges, edge_points, coords, i, j):
        """Add a line between the i-th and j-th points, if not in the list already"""
        if (i, j) in edges or (j, i) in edges:
            # already added
            return
        edges.add((i, j))
        edge_points.append(coords[[i, j]])

    coords = points

    tri = Delaunay(coords)
    edges = set()
    edge_points = []
    # loop over triangles:
    # ia, ib, ic = indices of corner points of the triangle
    for ia, ib, ic in tri.simplices:
        pa = coords[ia]
        pb = coords[ib]
        pc = coords[ic]

        # Lengths of sides of triangle
        a = math.sqrt((pa[0]-pb[0])**2 + (pa[1]-pb[1])**2)
        b = math.sqrt((pb[0]-pc[0])**2 + (pb[1]-pc[1])**2)
        c = math.sqrt((pc[0]-pa[0])**2 + (pc[1]-pa[1])**2)

        # Semiperimeter of triangle
        s = (a + b + c)/2.0

        # Area of triangle by Heron's formula
        area = math.sqrt(s*(s-a)*(s-b)*(s-c))
        circum_r = a*b*c/(4.0*area)

        # Here's the radius filter.
        # print circum_r
        if circum_r < 1.0/alpha:
            add_edge(edges, edge_points, coords, ia, ib)
            add_edge(edges, edge_points, coords, ib, ic)
            add_edge(edges, edge_points, coords, ic, ia)

    m = geometry.MultiLineString(edge_points)
    triangles = list(polygonize(m))
    return cascaded_union(triangles), edge_points, edges


def enveloppe_of_cluster(LD_data):

    # rescale LD_data to (0, 100),(0, 100)
    min = np.min(LD_data, axis=0)
    max = np.max(LD_data, axis=0)
    LD_data_rescale = LD_data - np.min(LD_data, axis=0)
    LD_data_rescale = LD_data_rescale / np.max(LD_data_rescale, axis=0) * 100

    concave_hull, edge_points, edges = alpha_shape(
        LD_data_rescale, alpha=0.001)

    xss = []
    yss = []

    indexs = []
    if concave_hull.geom_type == "Polygon":
        xs, ys = concave_hull.exterior.xy
        xs = np.array(xs)
        ys = np.array(ys)
        xs = xs / 100 * (max[0] - min[0]) + min[0]
        ys = ys / 100 * (max[1] - min[1]) + min[1]
        xss.append(xs)
        yss.append(ys)
        index = []
        for j in range(len(xs)):
            for k in range(len(LD_data)):
                if abs(xs[j] - LD_data[k][0]) < 0.01 and abs(ys[j] - LD_data[k][1]) < 0.01:
                    index.append(k)
                    break
        indexs.append(index)
    else:
        for polygon in concave_hull.geoms:
            xs, ys = polygon.exterior.xy
            xs = np.array(xs)
            ys = np.array(ys)
            xs = xs / 100 * (max[0] - min[0]) + min[0]
            ys = ys / 100 * (max[1] - min[1]) + min[1]
            xss.append(xs)
            yss.append(ys)
        # map each point of the polygone to the index in LD_data
        for i in range(len(xss)):
            index = []
            for j in range(len(xss[i])):
                for k in range(len(LD_data)):
                    if abs(xss[i][j] - LD_data[k][0]) < 0.01 and abs(yss[i][j] - LD_data[k][1]) < 0.01:
                        index.append(k)
                        break  # each point of the polygone is unique
            indexs.append(index)
    return xss, yss, indexs


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
