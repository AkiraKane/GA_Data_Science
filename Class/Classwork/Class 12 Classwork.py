# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 18:40:13 2013

@author: alexandersedgwick
"""

# Libraries and seed set
from pandas import DataFrame
from matplotlib import pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
np.random.seed(500)


recorders   = DataFrame({'locations' : ('A', 'B', 'C', 'D'), 'X' : (0, 0, 1, 1), 'Y' : (0, 1, 1, 0)})
locations   = np.array([ [.3, .5], [.8, .2] ])
intensities = np.array([
    [np.sin(np.array(range(100)) * np.pi/10) + 1.2],
    [np.cos(np.array(range(100)) * np.pi/15) * .7 + .9]]).T
distances   = np.array([
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



colors = ('red', 'blue', 'green', 'orange')
for i in range(4):
    plt.plot(range(100), pca.components_[i], c=colors[i])
    
    
    
    