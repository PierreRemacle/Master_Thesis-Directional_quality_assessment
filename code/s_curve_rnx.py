
from sklearn import manifold, datasets  # datasets
from nxcurve import quality_curve

from ipywidgets import interact, IntSlider
from mpl_toolkits.mplot3d import art3d
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA  # PCA
from sklearn.manifold import Isomap  # Isomap
from Utils import *
import sys
import numpy
import matplotlib.pyplot as pyplot
import matplotlib as matplotlib
from alive_progress import alive_bar
import sklearn
import sklearn.datasets

from matplotlib.widgets import Slider

n_comp = 2        # number of components to be reduced
n_nei = 20        # nearest neighbors
nsamples = 2000   # number of points (samples)

# Creating manifold
s_curve = sklearn.datasets.make_s_curve(
    n_samples=1000, noise=0.0, random_state=0)
print(s_curve)


# Performing dimensionality reduction
# PCA
pca = PCA(n_components=n_comp)
X_r = pca.fit_transform(s_curve[0])
# Isomap
iso = Isomap(n_neighbors=n_nei, n_components=n_comp)
X_r_2 = iso.fit_transform(s_curve[0])

# Drawing RNX curve
# quality_curve(s_curve[0], X_r, None,  'r', True)
# quality_curve(s_curve[0], X_r_2, None,  'b', True)
Q_PCA = quality_curve(s_curve[0], X_r, None, 'r')
Q_ISO = quality_curve(s_curve[0], X_r_2, None, 'r')
print(Q_PCA)
# Plotting
fig = pyplot.figure()
ax = fig.add_subplot(111)
ax.plot(Q_PCA[0], 'r')
ax.plot(Q_ISO[0], 'b')
ax.set_xlabel('Number of Neighbors')
ax.set_ylabel('100 * RNX')
ax.set_title('RNX Curve for S-Curve')
ax.legend(['PCA', 'Isomap'])
pyplot.show()
