
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


# load data

folder = 'PCA_COIL-20_data'
# load data
LD_data = np.load('./'+folder+'/LD_data.npy')
HD_data = np.load('./'+folder+'/HD_data.npy')


def delaunay_graph(LD_data):
    tri = Delaunay(LD_data)
    print(tri.simplices)
    grah = np.zeros((len(LD_data), len(LD_data)))
    for i in range(len(tri.simplices)):
        for j in range(3):
            grah[tri.simplices[i][j]][tri.simplices[i]
                                      [(j+1) % 3]] = np.linalg.norm(LD_data[tri.simplices[i][j]] - LD_data[tri.simplices[i][(j+1) % 3]])**2
            grah[tri.simplices[i][(j+1) % 3]][tri.simplices[i][j]
                                              ] = grah[tri.simplices[i][j]][tri.simplices[i][(j+1) % 3]]
    return grah
