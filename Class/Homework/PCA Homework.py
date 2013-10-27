# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 16:22:08 2013

@author: alexandersedgwick
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 07:52:35 2013
@author: asedgwick
"""
# Libraries and seed set
from pandas import DataFrame
from matplotlib import pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
np.random.seed(500)
recorders = DataFrame({'locations' : ('A', 'B', 'C', 'D'), 'X' : (0, 0, 1, 1), 'Y' : (0, 1, 1, 0)})
locations = np.array([ [.3, .5], [.8, .2] ])
intensities = np.array([
[np.sin(np.array(range(100)) * np.pi/10) + 1.2],
[np.cos(np.array(range(100)) * np.pi/15) * .7 + .9]]).T
distances = np.array([
np.sqrt((locations[0] - recorders.X[i])**2 + (locations[1] - recorders.Y[i])**2) for i in range(4)]).T
data = np.dot(intensities, np.exp(-2*distances))
data_transposed = data.T
row_means = [np.mean(i) for i in data_transposed]
data_transposed_scaled = np.array([data_transposed[i][0] - row_means[i] for i in range(4)])
pca = PCA()
pca.fit(data_transposed_scaled)
variance = pca.explained_variance_ratio_
readable_variance = variance * (1/variance[0])
plt.plot(range(4), readable_variance)
plt.show()
#
#colors = ('red', 'blue', 'green', 'orange')
#for i in range(4):
# plt.plot(range(100), pca.components_[i], c=colors[i])
#
#PCA for iris dataset
import pylab as pl
from sklearn.decomposition import PCA
from sklearn.lda import LDA
from sklearn.datasets import load_iris
from matplotlib import pyplot as plt2
iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names
pca = PCA()
X_r = pca.fit(X).transform(X)
variance = pca.explained_variance_ratio_
readable_variance = variance * (1/variance[0])
plt.plot(range(4), readable_variance)
plt.show()
iris.feature_names
pca.components_
#Kernel PCA for iris dataset
kpca = KernelPCA(kernel="rbf fit_inverse_transform=True)
X_kpca = kpca.fit_transform(X)
X_back = kpca.inverse_transform(X_kpca)
pca = PCA()
X_pca = pca.fit_transform(X)
# Plot results
reds = y == 0
blues = y == 1
pl.figure()
pl.title("Original space")
pl.plot(X[reds, 0], X[reds, 1], "ro")
pl.plot(X[blues, 0], X[blues, 1], "bo")
pl.xlabel("$x_1$")
pl.ylabel("$x_2$")
pl.show()
pl.figure()
pl.plot(X_pca[reds, 0], X_pca[reds, 1], "ro")
pl.plot(X_pca[blues, 0], X_pca[blues, 1], "bo")
pl.title("Projection by PCA")
pl.xlabel("1st principal component")
pl.ylabel("2nd component")
pl.show()
pl.figure()
pl.plot(X_kpca[reds, 0], X_kpca[reds, 1], "ro")
pl.plot(X_kpca[blues, 0], X_kpca[blues, 1], "bo")
pl.title("Projection by KPCA")
pl.xlabel("1st principal component in space induced by $\phi$")
pl.ylabel("2nd component")
pl.show()



#Wine KPCA

import numpy as np
import pylab as pl
import pandas as pd
wine = pd.read_csv('C:\Wine.csv')

from sklearn.decomposition import PCA, KernelPCA

X = wine.iloc[:,1:]
y =wine.iloc[:,0]

reds = y == 0
blues = y == 1
greens = y == 2
yellows = y == 3

kpca = KernelPCA(kernel="rbf", fit_inverse_transform=True, gamma=10)
X_kpca = kpca.fit_transform(X)
X_back = kpca.inverse_transform(X_kpca)

pca = PCA()
X_pca = pca.fit_transform(X)

pl.figure()
#pl.subplot(2, 2, 1, aspect='equal')

pl.plot(X_kpca[reds, 0], X_kpca[reds, 1], "ro")
pl.plot(X_kpca[blues, 0], X_kpca[blues, 1], "bo")
pl.plot(X_kpca[greens, 0], X_kpca[greens, 1], "go")
pl.plot(X_kpca[yellows, 0], X_kpca[yellows, 1], "yo")

pl.title("Projection by KPCA")
pl.xlabel("1st principal component in space induced by $\phi$")
pl.ylabel("2nd component")


pl.figure()
#pl.subplot(2, 2, 2, aspect='equal')
pl.plot(X_pca[reds, 0], X_pca[reds, 1], "ro")
pl.plot(X_pca[blues, 0], X_pca[blues, 1], "bo")
pl.plot(X_pca[greens, 0], X_pca[greens, 1], "go")
pl.plot(X_pca[yellows, 0], X_pca[yellows, 1], "yo")
pl.title("Projection by PCA")
pl.xlabel("1st principal component")
pl.ylabel("2nd component")

pl.show()
    
    