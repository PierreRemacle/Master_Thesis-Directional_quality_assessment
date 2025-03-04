from Utils import *
import sys
import numpy
import matplotlib.pyplot as pyplot
import matplotlib as matplotlib
from alive_progress import alive_bar
from numba import jit
import argparse
import pandas as pd
import numpy as np

from sklearn import datasets
from sklearn.manifold import TSNE, Isomap, MDS

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Quality of path")
    parser.add_argument("-HD", type=str, help="HD data", default=None)
    parser.add_argument("-LD", type=str, help="LD data", default=None)
    parser.add_argument("-QNX", type=bool, help="display QNX", default=False)
    args = parser.parse_args()
    if args.HD != None and args.LD != None:
        HD_data = pd.read_csv(
            "../DATA/" + args.HD + "HD.csv").to_numpy()[:, [1, 2]]
        LD_data = pd.read_csv(
            "../DATA/" + args.LD + "LD.csv").to_numpy()[:, [1, 2]]

    mnist = datasets.fetch_openml('mnist_784')
    # only 3000 samples are used for performance reasons
    mnist.data = mnist.data[:3000]
    mnist.target = mnist.target[:3000]
    HD_data = mnist.data / 255.0  # Normalize
    HD_data = np.array(HD_data, dtype=float)
    # tsne with 42 as seed
    LD_data = TSNE(n_components=2, random_state=42).fit_transform(HD_data)
    print(HD_data)
    # results, localisation_of_errors = ALL_path_2(
    #     HD_data, LD_data)

    # graph = delaunay_graph(LD_data)
    LD_data_reshaped = (LD_data - np.min(LD_data)) / \
        (np.max(LD_data) - np.min(LD_data)) * 100
    _, _, edges = alpha_shape(LD_data_reshaped, alpha=1)

    distance_matrix_LD = compute_distance_matrix(LD_data)
    graph = distance_matrix_LD.copy()
    graph = graph**2 + np.max(graph) * 100
    for i in range(len(graph)):
        graph[i][i] = 0  # remove self loop
    for edge in edges:
        graph[edge[0]][edge[1]] = distance_matrix_LD[edge[0]][edge[1]]**2
        graph[edge[1]][edge[0]] = distance_matrix_LD[edge[1]][edge[0]]**2
    # results, localisation_of_errors, error_per_start = random_selection_path_2(
    #     HD_data, LD_data, 200, graph)
    xss, yss, index_enveloppes = enveloppe_of_cluster(LD_data)
    # flatten the list
    index_enveloppe = [
        item for sublist in index_enveloppes for item in sublist]
    # results, localisation_of_errors, error_per_start = ALL_path_enveloppe_HDcompare(
    #     HD_data, LD_data, graph, index_enveloppe)
    results, localisation_of_errors, error_per_start = ALL_path_enveloppe(
        HD_data, LD_data, graph, index_enveloppe)
    average = {}
    for key in results:
        average[key] = sum(results[key]) / (len(results[key]))
    average = dict(sorted(average.items()))
    if args.QNX:
        plt = make_subplots(rows=3, cols=2)
    else:
        plt = make_subplots(rows=4, cols=2)

    x = np.sort(np.array(list(average.keys())))
    y_values = np.array(list(average.values()))
    # save the data
    np.savetxt("mnist_out.csv", np.array([x, y_values]).T, delimiter=",")

    plt.add_trace(go.Scatter(x=x, y=y_values, mode='lines',
                  name='base'), row=2, col=1)

    x_steps = np.asarray(list(average.keys()))

    y_diago = (x_steps-2)/(x_steps)
    standerized_x = (x_steps-2) / np.max(x_steps-2) * 100
    my_rnx = 1-(y_values / y_diago)
    # x_steps = np.asarray(list(average.keys()))
    #
    # y_diago = (x_steps-2)
    # standerized_x = (x_steps-2) / np.max(x_steps-2) * 100
    # my_rnx = ((x_steps-2) - y_values)/(x_steps-2)
    my_rnx[0] = 1
    print("overall score : ", sum(my_rnx) / len(my_rnx))
    plt.add_trace(go.Scatter(x=standerized_x, y=list(average.values()), mode='lines',
                             name='data'), row=3, col=1)
    plt.add_trace(go.Scatter(x=standerized_x, y=y_diago*x_steps, mode='lines',
                             name='esperance'), row=2, col=2)
    plt.add_trace(go.Scatter(x=standerized_x, y=y_values*x_steps, mode='lines',
                             name='esperance'), row=2, col=2)

    plt.add_trace(go.Scatter(x=standerized_x, y=y_diago, mode='lines',
                             name='esperance'), row=3, col=1)
    plt.add_trace(go.Scatter(x=standerized_x, y=my_rnx, mode='lines',
                             name='my_rnx'), row=3, col=2)

    plt.add_trace(go.Scatter(x=LD_data[:, 0], y=LD_data[:, 1], mode='markers',
                             name='lines+markers'), row=1, col=1)
    plt.add_trace(go.Scatter(x=LD_data[:, 0], y=LD_data[:, 1], mode='markers',
                             name='lines+markers'), row=1, col=2)
    if args.QNX:
        with alive_bar(1) as bar:
            qnxdata = [[], []]
            for i in range(1, len(LD_data)+1):
                qnxdata[0].append(i)
                qnxdata[1].append(QNX(LD_data, HD_data, i))
            plt.add_trace(go.Scatter(x=qnxdata[0], y=qnxdata[1], mode='lines',
                                     name='QNX'), row=4, col=1)
            rnxdata = [[], []]
            for i in range(1, len(LD_data)-1):
                rnxdata[0].append(i)
                rnxdata[1].append(RNX(LD_data, HD_data, i))
            bar()
        plt.add_trace(go.Scatter(x=rnxdata[0], y=rnxdata[1], mode='lines',
                                 name='RNX'), row=4, col=2)

    plt.update_xaxes(title_text='Lavenshtein distance', row=2, col=1)

    plt.update_xaxes(title_text='Lavenshtein distance - average', row=2, col=2)
    if args.QNX:
        plt.update_xaxes(title_text='QNX', row=4, col=1)
        plt.update_xaxes(title_text='RNX', row=4, col=2)
    for i in range(len(localisation_of_errors)):
        for j in range(len(localisation_of_errors[i])):
            if localisation_of_errors[i][j] > 0:
                cmap = pyplot.get_cmap('RdYlGn_r')
                color = cmap(
                    localisation_of_errors[i][j] / localisation_of_errors.max())

                color_hex = matplotlib.colors.rgb2hex(color)
                plt.add_trace(go.Scatter(x=[LD_data[i][0], LD_data[j][0]], y=[LD_data[i][1], LD_data[j][1]],
                                         mode='lines', name='lines+markers', line={"color": color_hex}), row=2, col=2)

    errorAtNode = np.zeros(len(localisation_of_errors))
    for i in range(len(localisation_of_errors)):
        errorAtNode[i] = sum(localisation_of_errors[i]) + \
            sum(localisation_of_errors.T[i])

    plt.add_trace(go.Scatter(
        x=LD_data[:, 0], y=LD_data[:, 1], marker={'color': error_per_start, 'colorscale': 'RdYlGn_r'}, mode='markers', name='LD_data'), row=1, col=2)
    print(errorAtNode)
    # dont show legend
    plt.update_layout(showlegend=False)
    plt.show()
