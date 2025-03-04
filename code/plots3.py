import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
from Utils import *
# Load the data
my_rnx_list = []


for i, embedding_name in enumerate(['t-SNE', 'PCA', 'Isomap', 'UMAP']):
    LD_data = np.load('./'+embedding_name+'/LD_data.npy')
    HD_data = np.load('./'+embedding_name+'/HD_data.npy')
    results = np.load('./'+embedding_name+'/results.npy', allow_pickle=True)
    # reduce size of dotes
    ax = plt.subplot2grid((4, 2), (i // 2, i % 2))
    ax.scatter(LD_data[:, 0], LD_data[:, 1], label=embedding_name, s=1)
    ax.legend()
    ax.set_title(f'{embedding_name} Embedding')
    results = results.tolist()
    freq = {}
    total = 0
    for key in results:
        freq[key] = len(results[key])
        total += len(results[key])
# sort the dictionary
    freq = dict(sorted(freq.items()))
    for key in freq:
        freq[key] = freq[key]/total

    sumhere = 0
    for key in freq:
        sumhere += freq[key]
# key where the cumulative sum is greater than 0.95
    key_99 = 0
    for key in freq:
        sumhere -= freq[key]
        if sumhere < 0.01:
            key_99 = key
            break

    # for key in list(results.keys()):
    #     if key > key_99:
    #         results.pop(key)

    average = {}
    for key in results:
        results[key] = np.array(results[key])

        results[key] = [pair[0] for pair in results[key]]
        average[key] = sum(results[key]) / len(results[key])

    # average = dict(sorted(average.items()))
    #
    # x_steps = np.asarray(list(average.keys()))
    # # print(x_steps)
    # average_y = np.array(list(average.values()))
    # m = 1
    # b = -1.7  # found experimentally
    # diago = (x_steps+b)/x_steps
    #
    # # rescale diagonal to 0-1
    # average_y = (average_y - np.min(diago)) / (np.max(diago) - np.min(diago))
    # diago = (diago - np.min(diago)) / \
    #     (np.max(diago) - np.min(diago))
    #
    # pathrnx = (diago - average_y)/diago
    average = dict(sorted(average.items()))

    x = np.sort(np.array(list(average.keys())))
    y_values = np.array(list(average.values()))
    x_steps = np.asarray(list(average.keys()))
    y_diago = (x_steps-1.7)
    y_values = (y_values - np.min(y_diago)) / \
        (np.max(y_diago) - np.min(y_diago))
    y_diago = (y_diago - np.min(y_diago)) / \
        (np.max(y_diago) - np.min(y_diago))
    my_rnx = y_diago - y_values
    my_rnx_list.append(my_rnx)
# Combine subplots 23 and 24 for the RNX plot
ax_rnx = plt.subplot2grid((4, 2), (2, 0), colspan=2, rowspan=2)

# Plot the RNX values for each embedding
ax_rnx.plot(np.array(
    range(len(my_rnx_list[0])))/len(my_rnx_list[0]), my_rnx_list[0], label='t-SNE')
ax_rnx.plot(np.array(
    range(len(my_rnx_list[1])))/len(my_rnx_list[1]), my_rnx_list[1], label='PCA')
ax_rnx.plot(np.array(
    range(len(my_rnx_list[2])))/len(my_rnx_list[2]), my_rnx_list[2], label='Isomap')
ax_rnx.plot(np.array(
    range(len(my_rnx_list[3])))/len(my_rnx_list[3]), my_rnx_list[3], label='UMAP')


# Customize the RNX plot
ax_rnx.set_title("RNX Comparison")
ax_rnx.legend()
ax_rnx.set_xlabel("Path Length")
ax_rnx.set_ylabel("RNX Value")

# Adjust layout to prevent overlapping
plt.tight_layout()
plt.show()
