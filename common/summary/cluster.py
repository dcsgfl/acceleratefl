"""
cluster.py

Clustering algorithms for flsys
UMN DCSG, 2021
"""

import numpy as np
from numpy import unique

from sklearn.cluster import DBSCAN

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
def cluster_mat(matList, keySpace):

    pass

