import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Load the data
# data = pd.read_csv('../PLOTS/MNIST_Isomap_rnx.csv', header=None)
# data_new = pd.read_csv('../PLOTS/MNIST_Isomap_new_rnx.csv', header=None)
# data = data.to_numpy()
# data_new = data_new.to_numpy()
# print(data_new)

for method in ["PCA", "t-SNE", "UMAP", "Isomap", "MDS"]:
    data = pd.read_csv(
        f'../PLOTS/MNIST_{method}_new_rnx_all.csv', header=None)
    data = data.to_numpy()
    for i in range(len(data)):
        data[i] = data[i] + i
    # create a range from 0 to 1 with the same length as the data
    data_x = np.linspace(0, 1, len(data))
    plt.plot(data_x, data, label=method)

plt.legend()
# plt.xscale('log')
# plot the data

plt.show()
