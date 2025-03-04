from Utils import *
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from minisom import MiniSom  # SOM implementation
import umap
from sklearn.manifold import TSNE, Isomap, MDS
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
from sklearn import datasets
import numpy as np
from PIL import Image
import os
from concurrent.futures import ThreadPoolExecutor


folder = 'PCA_COIL-20_data'
# load data
LD_data = np.load('./'+folder+'/LD_data.npy')
HD_data = np.load('./'+folder+'/HD_data.npy')
