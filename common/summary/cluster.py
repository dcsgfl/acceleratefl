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

def hdist(v1, v2):

    sqrt2 = np.sqrt(2)

    sqrtdevP = np.sqrt(v1)
    sqrtdevQ = np.sqrt(v2)
    sumOfSqrOfDiffSqrtdevice = np.sum((sqrtdevP - sqrtdevQ) ** 2)
    return np.sqrt(sumOfSqrOfDiffSqrtdevice) / sqrt2

def mnorm(m1, m2):

    avgDist = 0.0
    for x in range(m1.shape[0]):
        d = hdist(m1[x,:], m2[x,:])
        avgDist += d
    return avgDist / float(m1.shape[0])

"""
Basic clustering routine for clustering histograms.
Searches for clusters of at least 2 points. Returns an array of assignments
with a -1 indicating that no suitable cluster was found for that device.
"""
def cluster_hist(histogramList, keySpace):

    X = np.zeros((len(histogramList), len(keySpace)))
    for idx, hist in enumerate(histogramList):
        X[idx,:] = hist.toFrequencyArray(keySpace)

    dim = len(histogramList)
    distMat = np.zeros((dim, dim))

    for i in range(dim):
        for j in range(i, dim):
            #h1 = histogramList[i]
            #h2 = histogramList[j]
            #dist = h1.computeHellingerDist(h2, keySpace)
            dist = hdist(X[i,:], X[j,:])
            distMat[i,j] = dist
            distMat[j,i] = dist

    #return DBSCAN(min_samples=2,eps=0.33,
    #              metric="precomputed").fit_predict(distMat)
    return OPTICS(min_samples=2,
                  metric="precomputed").fit_predict(distMat)

    #return OPTICS(min_samples=2).fit_predict(X)
    #return OPTICS(min_samples=2).fit_predict(X)
    #model = DBSCAN(eps=0.30, min_samples=2)
    #yhat = model.fit_predict(X)

    #return yhat

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
            #dist = norm(mats[i] - mats[j], 2)
            dist = mnorm(mats[i], mats[j])
            distMat[i,j] = dist
            distMat[j,i] = dist

    return OPTICS(min_samples=2,
                  metric="precomputed").fit_predict(distMat)
    #return DBSCAN(min_samples=2,eps=0.33,
    #              metric="precomputed").fit_predict(distMat)
    #model = DBSCAN(eps=1.13, min_samples=2,
    #               metric='precomputed')
    #yhat = model.fit_predict(distMat)
    #
    #return yhat

