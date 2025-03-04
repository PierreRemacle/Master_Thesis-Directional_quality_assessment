import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.datasets import make_classification
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import sys
from plotly.subplots import make_subplots


def gen_synthetic_data(n_samples=1500, n_features=20, n_classes=3, n_informative=8, n_clusters_per_class=1):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=0,
        n_classes=n_classes,
        n_clusters_per_class=n_clusters_per_class,

    )
    return X, y


def tsne(X, y):
    X_embedded = TSNE(n_components=2).fit_transform(X)
    return X_embedded


def pca(X, y):
    X_embedded = PCA(n_components=2).fit_transform(X)
    return X_embedded


def radom_reduction(X, y):
    X_embedded = np.random.rand(X.shape[0], 2)
    return X_embedded
# save data


def save_data(X, y, name):

    df = pd.DataFrame(X, y)
    df.to_csv("../DATA/" + name + ".csv")


if __name__ == "__main__":
    if len(sys.argv) != 7:
        print("Usage: python Generate_data.py n_samples n_features n_classes n_informative n_clusters_per_class name")
        sys.exit(1)

    n_samples = int(sys.argv[1])
    n_features = int(sys.argv[2])
    n_classes = int(sys.argv[3])
    n_informative = int(sys.argv[4])
    n_clusters_per_class = int(sys.argv[5])
    name = sys.argv[6]
    X, y = gen_synthetic_data(n_samples, n_features,
                              n_classes, n_informative, n_clusters_per_class)
    X_embedded_tsne = tsne(X, y)
    X_embedded_pca = pca(X, y)
    X_embedded_random = radom_reduction(X, y)
    save_data(X, y, str(name)+"HD")
    save_data(X_embedded_tsne, y, str(name)+"_tsne_LD")
    save_data(X_embedded_pca, y, str(name)+"_pca_LD")
    save_data(X_embedded_random, y, str(name)+"_random_LD")
    print("Data saved")
