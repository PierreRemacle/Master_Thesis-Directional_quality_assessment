from Utils import *
from sklearn import linear_model
from scipy.stats import poisson
import sys
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import itertools
import time
import Levenshtein as lev

knn, graph = KNN_graph(LD_data, 30)


# Create a figure with make_subplots
def display():

    result, walks = random_walk(knn, graph, nbr_itter=10, nbr_walk=100)
    fig = make_subplots(rows=1, cols=2)
    fig.add_trace(go.Bar(x=list(result.keys()), y=list(
        result.values()), name='barchart'), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=LD_data["0"], y=LD_data["1"], mode="markers"), row=1, col=2)
    # You can choose any color scale from Plotly Express
    color_scale = px.colors.qualitative.Plotly

    for i, walk in enumerate(walks):
        x_coords = [LD_data.iloc[idx]["0"] for idx in walk]
        y_coords = [LD_data.iloc[idx]["1"] for idx in walk]

        trace = go.Scatter(
            x=x_coords,
            y=y_coords,
            mode='lines',
            name='Walk ' + str(i),
            line=dict(color=color_scale[i % len(color_scale)])
        )

        fig.add_trace(trace, row=1, col=2)

    fig.show()


def checkforconvergence():
    fig = make_subplots(rows=13, cols=2)
    nbr = 50
    results = np.zeros((nbr, 22))

    for i in range(nbr):
        result, walks = random_walk(
            knn, graph, nbr_itter=20, nbr_walk=10)

        for j in range(nbr, i, -1):
            for key, value in result.items():
                results[j-1][int(key)] += value
        results[j-1] = results[j-1] / (j)
        print(i)

    # display the results of the last 10 iterations
    fig.add_trace(go.Scatter(x=[(i+1) * 10 for i in range(len(results[-1]))],
                  y=results[-1], mode='lines', name="len of " + str(i)), row=13, col=1)
    results = results.T

    for i in range(len(results)):
        fig.add_trace(go.Scatter(x=[(i+1) * 10 for i in range(len(results[i]))], y=results[i],
                                 mode='lines', name="len of " + str(i)), row=(i//2+1), col=(i % 2+1))
    fig.show()


def checkforconvergencelen():
    fig = make_subplots(rows=1, cols=1)

    result, walks = random_walk(knn, graph, nbr_itter=20, nbr_walk=200)
    Ls = []
    penalitys = []
    for data in walks:
        walk = data[0]
        penality = data[1]
        L = 0
        for i in range(len(walk)-1):
            L += np.linalg.norm(LD_data.iloc[walk[i]] -
                                LD_data.iloc[walk[i+1]])
        Ls.append(L)
        penalitys.append(penality)
    fig.add_trace(go.Scatter(x=Ls, y=penalitys,
                             mode='markers'), row=1, col=1)
    fig.show()


def expectedAverageLavenstein():
    results = []
    results_len = []
    for i in range(6):
        print(i+1)
        initial_list = np.array(range(i+1))
        results_len.append(i+1)
        compared_list = np.array(initial_list).copy()
        penality = 0
        dict_penality = {}
        for j in range(5000000):
            np.random.shuffle(compared_list)
            tempo = levenshteinDistanceDP(initial_list, compared_list)
            if dict_penality.get(tempo) is None:
                dict_penality[tempo] = 1
            else:
                dict_penality[tempo] += 1
            penality += tempo
        for key, value in dict_penality.items():
            dict_penality[key] = value * (np.math.factorial(i+1)) / 5000000
        print(dict_penality, np.math.factorial(i+1))
        penality = penality/5000000
        results.append(penality)
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Scatter(x=results_len, y=results,
                             mode='lines'), row=1, col=1)
    fig.show()


def trueexpectedaveragelavensteinrandom(size=5):
    start = time.time()
    init_list = np.array(range(size))
    dico_result = {}
    for i in range(size*10000):
        i = init_list.copy()
        np.random.shuffle(i)
        penality = levenshteinDistanceDP(init_list, i)
        if dico_result.get(penality) is None:
            dico_result[penality] = 1
        else:
            dico_result[penality] += 1
    weigted_average = 0
    for key, value in dico_result.items():
        weigted_average += key * value
    # savetofile
    stop = time.time()
    print(weigted_average / (size*10000))
    print("time for size " + str(size) + " : " + str(stop-start))
    with open("editdistanceesperance.txt", "a") as file:
        file.write(str(size) + " " + str(weigted_average /
                   (size*10000)) + "\n")


def trueexpectedaveragelavenstein(size=5):
    start = time.time()
    init_list = np.array(range(size))
    combinations = list(itertools.permutations(init_list))
    dico_result = {}
    for i in combinations:
        penality = levenshteindistancedp(init_list, i)
        if penality == size:
            print(i)
        if dico_result.get(penality) is none:
            dico_result[penality] = 1
        else:
            dico_result[penality] += 1
    weigted_average = 0
    for key, value in dico_result.items():
        weigted_average += key * value
    # savetofile
    stop = time.time()
    print("time for size " + str(size) + " : " + str(stop-start))
    with open("editdistanceesperance.txt", "a") as file:
        file.write(str(size) + " " + str(weigted_average /
                   np.math.factorial(size)) + "\n")


def randomlavensteinhisto(size=5):
    init_list = np.array(range(size))
    dico_result = {}
    for i in range(100000):
        i = init_list.copy()
        np.random.shuffle(i)
        penality = levenshteinDistanceDP(init_list, i)
        if dico_result.get(penality) is None:
            dico_result[penality] = 1
        else:
            dico_result[penality] += 1
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Bar(x=list(dico_result.keys()), y=list(
        dico_result.values()), name='barchart'), row=1, col=1)

    fig.show()


def RNX_plot():
    results_rnx = []
    results_qnx = []
    for k in range(1, len(LD_data)-1):
        rnx = RNX(LD_data, HD_data, k)
        results_rnx.append(rnx)
        qnx = QNX(LD_data, HD_data, k)
        results_qnx.append(qnx)
    fig = make_subplots(rows=1, cols=2)
    fig.add_trace(go.Scatter(x=list(range(1, len(LD_data)-1)),
                  y=results_rnx, mode='lines', name="RNX"), row=1, col=1)
    fig.add_trace(go.Scatter(x=list(range(1, len(LD_data)-1)),
                  y=results_qnx, mode='lines', name="QNX"), row=1, col=2)

    fig.show()


class Graph:
    def __init__(self, cost, size, path=["0"], position=(0, 0)):
        self.path = path
        self.cost = cost
        self.size = size
        self.position = position
        self.F = None
        self.R = None
        self.B = None
        self.D = None

    def get_all_position(self, l=[]):
        l.append(self.position)
        if self.R != 0:
            self.R.get_all_position(l)
        if self.F != 0:
            self.F.get_all_position(l)
        if self.B != 0:
            self.B.get_all_position(l)
        if self.D != 0:
            self.D.get_all_position(l)

        return l

    def all_path(self, l=[], cost=[]):
        if self.R != 0:
            self.R.all_path(l)
        if self.F != 0:
            self.F.all_path(l)
        if self.B != 0:
            self.B.all_path(l)
        if self.D != 0:
            self.D.all_path(l)
        if self.R == 0 and self.F == 0 and self.B == 0 and self.D == 0 and self.cost == 0:
            l.append(self.path)
            cost.append(self.cost)
        return l, cost


def GraphExplore(cost=2, size=4):
    graph = Graph(cost, size)

    return GraphExploreRecursive(graph)


def GraphExploreRecursive(graph):
    cost = graph.cost
    size = graph.size
    # go right cost of 1
    if cost > 0 and np.abs(graph.position[1]-(graph.position[0] + 1)) < cost and graph.position[0] < size and graph.path[-1] != "B":
        graph.R = Graph(cost-1, size, graph.path + ["R"],
                        (graph.position[0] + 1, graph.position[1]))
        GraphExploreRecursive(graph.R)
    else:
        graph.R = 0
    # go down cost of 1
    if cost > 0 and np.abs((graph.position[1] + 1)-graph.position[0]) < cost and graph.position[1] < size and graph.path[-1] != "R":
        graph.B = Graph(cost-1, size, graph.path +
                        ["B"], (graph.position[0], graph.position[1] + 1))
        GraphExploreRecursive(graph.B)
    else:
        graph.B = 0
    # go diagonal cost of 1
    if cost > 0 and graph.position[0] < size and graph.position[1] < size and np.abs(graph.position[1] - graph.position[0]) < cost:
        graph.D = Graph(cost-1, size, graph.path +
                        ["D"], (graph.position[0] + 1, graph.position[1] + 1))
        GraphExploreRecursive(graph.D)
    else:
        graph.D = 0
    # go diagonal but free
    if graph.position[0] < size - cost and graph.position[1] < size - cost:
        graph.F = Graph(cost, size, graph.path +
                        ["F"], (graph.position[0] + 1, graph.position[1] + 1))
        GraphExploreRecursive(graph.F)
    else:
        graph.F = 0
    return graph


print(delaunay_graph(LD_data))
print(levenshteinDistanceDP(
    [32, 7, 9, 4, 3, 14, 98], [7, 9, 32, 4, 98, 3, 14]))
print(lev.distance([32, 7, 9, 4, 3, 14, 98], [7, 9, 32, 4, 98, 3, 14]))
