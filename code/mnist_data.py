import plotly.graph_objects as go
import matplotlib.pyplot as plt
from keras.datasets import mnist
import pandas as pd
import numpy as np
import os
# tsne
from sklearn.manifold import TSNE
# pca
from sklearn.decomposition import PCA
# mds
from sklearn.manifold import MDS
# CCA
from sklearn.cross_decomposition import CCA
# SNE
from sklearn.manifold import SpectralEmbedding
(train_X, train_y), (test_X, test_y) = mnist.load_data()
# flatten 28*28 images to a 784 vector for each image
name = "mnist"
train_X = train_X.reshape(train_X.shape[0], 784)
print(train_X.shape)

# keep 3000 random selected images
np.random.shuffle(train_X)
train_X = train_X[:3000]

# tsne
print("tsne")
X_tsne = TSNE(n_components=2).fit_transform(train_X)

# PCA
print("pca")
X_pca = PCA(n_components=2).fit_transform(train_X)

# cmds
print("cmds")
X_cmds = MDS(n_components=2, metric=True,
             normalized_stress="auto").fit_transform(train_X)

# nmds
print("nmds")
X_nmds = MDS(n_components=2, metric=False,
             normalized_stress="auto").fit_transform(train_X)

# sne
print("sne")
X_sne = SpectralEmbedding(n_components=2).fit_transform(train_X)

# display
fig = go.Figure()
fig.add_trace(go.Scatter(x=X_tsne[:, 0],
              y=X_tsne[:, 1], mode='markers', name='tsne'))
fig.show()

# create new floder
if not os.path.exists("../DATA/"+name):
    os.mkdir("../DATA/"+name)
else:
    os.system("rm -r ../DATA/"+name)
    os.mkdir("../DATA/"+name)

# save data
df1 = pd.DataFrame(X_tsne)
df1.to_csv("../DATA/"+name+"/tsne.csv")
df2 = pd.DataFrame(X_pca)
df2.to_csv("../DATA/"+name+"/pca.csv")
df3 = pd.DataFrame(X_cmds)
df3.to_csv("../DATA/"+name+"/cmds.csv")
df4 = pd.DataFrame(X_nmds)
df4.to_csv("../DATA/"+name+"/nmds.csv")
df5 = pd.DataFrame(X_sne)
df5.to_csv("../DATA/"+name+"/sne.csv")
print("Data saved")
