"""
cluster.py

Clustering algorithms for flsys
UMN DCSG, 2021
"""

import numpy as np
from numpy import unique
from numpy.linalg import norm

from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS

"""
Basic clustering routine for clustering histograms.
Searches for clusters of at least 2 points. Returns an array of assignments
with a -1 indicating that no suitable cluster was found for that device.
"""
def cluster_hist(histogramList, keySpace):

    X = np.zeros((len(histogramList), len(keySpace)))
    for idx, hist in enumerate(histogramList):
        X[idx,:] = hist.toFrequencyArray(keySpace)

    model = DBSCAN(eps=0.30, min_samples=2)
    yhat = model.fit_predict(X)

    return yhat


"""
Basic clustering routine for clustering a list of histograms.
Searches for clusters of at least 2 points. Returns an array of assignments
with a -1 indicating that no suitable cluster was found for that device.
"""
def cluster_mat(matList, xKeySpace, yKeySpace):

    dim = len(matList)
    distMat = np.zeros((dim, dim))

    mats = []
    for i in range(dim):
        mat = matList[i].toMatrix(xKeySpace, yKeySpace)
        mats.append(mat)

    for i in range(dim):
        for j in range(i, dim):
            dist = norm(mats[i] - mats[j])
            distMat[i,j] = dist
            distMat[j,i] = dist

    return OPTICS(min_samples=2,
                  metric="precomputed").fit_predict(distMat)
    #model = DBSCAN(eps=1.13, min_samples=2,
    #               metric='precomputed')
    #yhat = model.fit_predict(distMat)
    #
    #return yhat

