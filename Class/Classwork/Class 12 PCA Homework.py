# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 20:19:23 2013

@author: alexandersedgwick
"""


from sklearn import datasets

iris = datasets.load_iris()
X, y = iris.data, iris.target


from sklearn.decomposition import PCA
pca = PCA(n_components=5, whiten=True).fit(X)

pca = PCA.fit(X)


pca.components_.T
pca.explained_variance_ratio_

pca.explained_variance_ratio_.sum()

X_pca = pca.transform(X)
X_pca.mean(axis=0)

X_pca.std(axis=0)
import numpy as np
np.corrcoef(X_pca.T)


variance = pca.explained_variance_ratio_
readable_variance = variance * (1/variance[0])
plt.plot(range(3), readable_variance)
plt.show()
