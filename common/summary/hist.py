"""
hist.py

Histogram summary implementation for flsys
UMN DCSG, 2021
"""

import numpy as np
import json

from collections import Counter

class HistSummary(object):

    """
    @param labels   List of all labels in the local training data.
                    A histogram will be constructed by counting the
                    instances of each label in the list.
    """
    def __init__(self, labels):

        if not isinstance(labels, list):
            raise ValueError("Labels must be a list")

        self.values = Counter(labels)

    """
    @brief Adds noise to the histogram satisfying (epsilon, 0) - DP
    """
    def addNoise(self, epsilon):
        
        for k in self.values.keys():
            self.values[k] += float(np.random.laplace(0, 1/epsilon, 1))

            if self.values[k] < 0.0:
                # Just say there is one so Jaccard Similarity can use it
                self.values[k] = 1

    def getKeys(self):
        return set(self.values.keys())

    def computeJaccardSimilarity(self, hist2):

        if not isinstance(hist2, HistSummary):
            raise ValueError("Input must be a HistSummary")

        keys1 = set(self.values.keys())
        keys2 = set(hist2.values.keys())

        return float(len(keys1.intersection(keys2))) / float(len(keys1.union(keys2)))

    def computeHellingerDist(self, hist2):

        if not isinstance(hist2, HistSummary):
            raise ValueError("Input must be a HistSummary")

        sqrt2 = np.sqrt(2)
        allKeys = (Counter(self.values) + Counter(hist2.values)).keys()

        devP = np.zeros(len(allKeys))
        devQ = np.zeros(len(allKeys))

        idx = 0
        for k in allKeys:
            
            if k in self.values:
                devP[idx] = float(self.values[k])

            if k in hist2.values:
                devQ[idx] = float(hist2.values[k])

            idx = idx + 1

        sqrtdevP = np.sqrt(devP)
        sqrtdevQ = np.sqrt(devQ)
        sumOfSqrOfDiffSqrtdevice = np.sum((sqrtdevP - sqrtdevQ) ** 2)
        return np.sqrt(sumOfSqrOfDiffSqrtdevice) / sqrt2

    """
    Serialization functions
    """
    def toJson(self):
        return json.dumps(self.values)

    def fromJson(self, jsonStr):
        self.values = json.loads(jsonStr)

    def __str__(self):
        return self.toJson()

    def toArray(self, keySpace):

        arr = np.zeros(len(keySpace))
        arrIdx = 0

        for key in keySpace:
            if key in self.values.keys():
                arr[arrIdx] = float(self.values[key])
            arrIdx += 1

        return arr

    def toFrequencyArray(self, keySpace):

        
        arr = self.toArray(keySpace)
        denom = 0.0
        for key in self.values.keys():
            denom += float(self.values[key])

        return arr / denom
