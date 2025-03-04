import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors
from Utils import *
import random
from matplotlib.animation import FuncAnimation


def path_rnx_distance_sorted(LD_data, HD_data, LD_paths, HD_paths):

    results = {}

    for index in range(len(LD_data)):

        results[index] = {}

        for i in range(len(LD_paths[index])):

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

            if len(LD_path) not in results[index]:
                results[index][len(LD_path)] = [distance]
            else:
                results[index][len(LD_path)].append(distance)
        # sort the keys
        results[index] = dict(sorted(results[index].items()))
        results[index] = {k: np.mean(v) for k, v in results[index].items()}

    with open('results_color.pkl', 'wb') as fp:
        pickle.dump(results, fp)


def rescale(LD_data, HD_data, LD_paths, HD_paths, axbig2, axbig):

    longest_path_len = 0
    ys = np.zeros([len(LD_data), 500])
    results_2 = [0] * len(LD_data)
    all_index_enveloppe = range(len(LD_data))
    # len of the longest path
    with open('results_color.pkl', 'rb') as fp:
        results = pickle.load(fp)

    for index in all_index_enveloppe:
        data = results[index]
        keys = list(data.keys())
        max_key = max(keys)
        if max_key > longest_path_len:
            longest_path_len = max_key

    means = np.zeros(longest_path_len)
    colors = np.zeros(len(LD_data))
    # for index in all_index_enveloppe:
    for index in [0, 1]:
        print(index)

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
        length = len(results[index])

        y = np.array(list(data.values()))/list(data.keys()) - \
            ((np.array(list(data.keys())) - 1.7)/(np.array(list(data.keys()))))
        # y sould have a length of 500
        ys[index] = np.pad(-y, (0, 500-len(y)), 'constant')
        # plt.plot(list(data.keys()), -y, alpha=0.05)
        if index % 1 == 0:
            ys_transpose = ys[:index].T

            means = np.zeros(longest_path_len)
            for i in range(longest_path_len):

                tempo = np.trim_zeros(np.sort(ys_transpose[i]))
                means[i] = np.mean(tempo)
            axbig2.clear()
            axbig2.plot(np.arange(2, len(means)+2),
                        means)
            for i in range(index):
                colors[i] = np.mean(ys[i][:len(means)] - means)
            y2 = 1 - ((np.arange(longest_path_len) - 1.7) /
                      (np.arange(longest_path_len)))

            y3 = 0.8 - ((np.arange(longest_path_len) - 1.7) /
                        (np.arange(longest_path_len)))
            y4 = 0.6 - ((np.arange(longest_path_len) - 1.7) /
                        (np.arange(longest_path_len)))
            y5 = 0.4 - ((np.arange(longest_path_len) - 1.7) /
                        (np.arange(longest_path_len)))
            y6 = 0.2 - ((np.arange(longest_path_len) - 1.7) /
                        (np.arange(longest_path_len)))
            y7 = 0.9 - ((np.arange(longest_path_len) - 1.7) /
                        (np.arange(longest_path_len)))
            y8 = 0.1 - ((np.arange(longest_path_len) - 1.7) /
                        (np.arange(longest_path_len)))
            y9 = 0.5 - ((np.arange(longest_path_len) - 1.7) /
                        (np.arange(longest_path_len)))
            y10 = 0.3 - ((np.arange(longest_path_len) - 1.7) /
                         (np.arange(longest_path_len)))
            y11 = 0.7 - ((np.arange(longest_path_len) - 1.7) /
                         (np.arange(longest_path_len)))
            y12 = 0 - ((np.arange(longest_path_len) - 1.7) /
                       (np.arange(longest_path_len)))

            axbig2.plot(np.arange(len(means)), -y2,
                        linestyle='dashed', c="grey", alpha=0.5)
            axbig2.plot(np.arange(len(means)), -y3,
                        linestyle='dashed', c="grey", alpha=0.5)
            axbig2.plot(np.arange(len(means)), -y4,
                        linestyle='dashed', c="grey", alpha=0.5)
            axbig2.plot(np.arange(len(means)), -y5,
                        linestyle='dashed', c="grey", alpha=0.5)
            axbig2.plot(np.arange(len(means)), -y6,
                        linestyle='dashed', c="grey", alpha=0.5)
            axbig2.plot(np.arange(len(means)), -y7,
                        linestyle='dashed', c="grey", alpha=0.5)
            axbig2.plot(np.arange(len(means)), -y8,
                        linestyle='dashed', c="grey", alpha=0.5)
            axbig2.plot(np.arange(len(means)), -y9,
                        linestyle='dashed', c="grey", alpha=0.5)
            axbig2.plot(np.arange(len(means)), -y10,
                        linestyle='dashed', c="grey", alpha=0.5)
            axbig2.plot(np.arange(len(means)), -y11,
                        linestyle='dashed', c="grey", alpha=0.5)
            axbig2.plot(np.arange(len(means)), -y12,
                        linestyle='dashed', c="grey", alpha=0.5)

            # y between 0 and 1
            axbig2.set_ylim(0, 1)
            axbig2.text(0, 0.1, str(index))

            axbig.clear()
            zero_mask = (colors == 0)
            non_zero_mask = ~zero_mask

            axbig.scatter(LD_data[zero_mask, 0],
                          LD_data[zero_mask, 1], c=colors[zero_mask])

            axbig.scatter(
                LD_data[non_zero_mask, 0], LD_data[non_zero_mask, 1], c=colors[non_zero_mask])

            plt.pause(0.01)

    # color each point according to the value in results_2

    # plt.scatter(LD_data[:, 0], LD_data[:, 1], c=np.log(results_2))
    # plt.legend()

    # plt.show()
    # path_rnx_distance_sorted(
    #     LD_data, HD_data, LD_paths_2, HD_paths_2)
    return means


def update(frame):
    # updating the data by adding one more point
    x.append(random.randint(1, 100))
    y.append(random.randint(1, 100))

    axbig2.clear()  # clearing the axes
    # creating new scatter chart with updated data
    axbig2.scatter(x, y, s=y, c='b', alpha=0.5)
    fig.canvas.draw()  # forcing the artist to redraw itself


fig, axs = plt.subplots(3, 2)
# combine axs 0,0 ; 1,0 ; 0,1 ; 1,1 into one plot
gs = axs[0, 0].get_gridspec()
for ax in axs[0:2, 0:2].flatten():
    ax.remove()

axbig = fig.add_subplot(gs[0:2, 0:2])
# combine axs 2,0 ; 2,1 into one plot
gs = axs[2, 0].get_gridspec()
for ax in axs[2, :].flatten():
    ax.remove()

axbig2 = fig.add_subplot(gs[2, :])


folder = 't-SNE_MNIST_data'
print(folder)
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

# path_rnx_distance_sorted(LD_data, HD_data, LD_paths_2, HD_paths_2)
plt.ion()  # turning interactive mode on
rescale(LD_data, HD_data, LD_paths_2, HD_paths_2, axbig2, axbig)
fig.tight_layout()
plt.show(block=True)
