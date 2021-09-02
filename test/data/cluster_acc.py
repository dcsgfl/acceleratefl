#!/usr/bin/env python3

import os
import sys

# Add data folder to path
pwd = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..', 'data')
sys.path.append(pwd)

from datasetFactory import DatasetFactory as dftry

pwd = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..', 'common','summary')
sys.path.append(pwd)

from cluster import cluster_hist
from hist import HistSummary
import numpy as np

NEXT_DEV_ID = 1

def getNextId():

    global NEXT_DEV_ID
    retVal = str(NEXT_DEV_ID)
    NEXT_DEV_ID += 1
    return retVal

def generatePy(datacls):

    devid = getNextId()
    _, _ = datacls.get_training_data(devid)
    _, _ = datacls.get_testing_data(devid)

    tensor_train_x, tensor_train_y = datacls.get_training_data(devid)
    train_y = tensor_train_y.numpy()

    counts = np.bincount(train_y)
    target = str(np.argmax(counts))

    histInput = list(map(str, train_y.tolist()))
    return devid, target, HistSummary(histInput)

def genDevices(count, eps=1.0):

    devs = {}
    labels = set()

    for i in range(count):

        datacls = dftry.getDataset("MNIST")
        datacls.download_data()

        devid, target, hs = generatePy(datacls)
        devs[devid] = [target, hs]

        labels = labels.union(hs.getKeys())

    return labels, devs

def addNoise(devs, eps):

    for key in devs:
        devs[key][1].addNoise(eps)

def getTrueClusters(devs, labelSpace):

    clust = {}

    for label in labelSpace:
        clust[label] = []

    for devid in devs.keys():
        dev = devs[devid]
        clust[dev[0]].append(devid)

    return clust

def predictClusters(devs, labels):

    hlist = []
    for devid in devs.keys():
        hlist.append(devs[devid][1])

    classes = cluster_hist(hlist, labels)
    return classes

def computeAcc(devs, trueClust, predClust):

    cids = set(predClust)
    devids = list(devs.keys())
    total = float(len(devids))
    correct = 0.0

    for cid in cids:
        
        matches = []
        for idx, tmpCID in enumerate(predClust):
            if tmpCID == cid:
                matches.append(devids[idx])

        devid = matches[0]
        expectedDevs = []

        # Find the true cluster
        for tc in trueClust.keys():
            if devid in trueClust[tc]:
                # We have a match
                expectedDevs = trueClust[tc]

        if cid == -1:

            # Need to manually verify each device
            for d in matches:
                for tc in trueClust.keys():
                    if d in trueClust[tc] and len(trueClust[tc]) == 1:
                        correct += 1.0

        elif sorted(matches) == sorted(expectedDevs):
            prop = float(len(set(matches).intersection(set(expectedDevs))))
            correct += (prop * 1.0)

    return correct / total

###
### Main logic
###

labels, devs = genDevices(40)
addNoise(devs, 1.0)
addNoise(devs, 0.001)
for dev in devs:
    print(devs[dev][0])
    print(devs[dev][1].toJson())

clust = getTrueClusters(devs, labels)
pred = predictClusters(devs, labels)
print(clust)
print(pred)

acc = computeAcc(devs, clust, pred)
print("Accuracy: " + str(acc * 100) + "%")

