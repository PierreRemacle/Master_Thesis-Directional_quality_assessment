
from _plotly_utils.utils import levenshtein
import numpy as np
import pandas as pd
from Utils import *
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay


# Load data

folder = 'PCA_mnist_data'
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

triangulation_hd = Delaunay(HD_data)
