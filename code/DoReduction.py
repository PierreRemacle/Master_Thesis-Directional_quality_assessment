import argparse
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
import pandas as pd
import os
import numpy as np
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="apply all dimension reduction")
    parser.add_argument("name", help="name of the dataset")
    args = parser.parse_args()
    name = args.name
    data = pd.read_csv("../DATA/" + name + ".csv").drop(columns="Unnamed: 0")
    X = data.to_numpy()
    print("tsne")
    X_tsne = TSNE(n_components=2).fit_transform(X)
    print("pca")
    X_pca = PCA(n_components=2).fit_transform(X)
    print("cmds")
    X_cmds = MDS(n_components=2, metric=True,
                 normalized_stress="auto").fit_transform(X)
    print("nmds")
    X_nmds = MDS(n_components=2, metric=False,
                 normalized_stress="auto").fit_transform(X)
    print("sne")
    X_sne = SpectralEmbedding(n_components=2).fit_transform(X)

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
