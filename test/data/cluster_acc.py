#!/usr/bin/env python3

import copy
import os
import sys

# Add data folder to path
pwd = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..', 'data')
sys.path.append(pwd)

from datasetFactory import DatasetFactory as dftry

pwd = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..', 'common','summary')
sys.path.append(pwd)

from cluster import cluster_hist
from cluster import cluster_mat

from hist import HistSummary
from hist import HistMatSummary

import numpy as np

NEXT_DEV_ID = 1

def getNextId():

    global NEXT_DEV_ID
    retVal = str(NEXT_DEV_ID)
    NEXT_DEV_ID += 1
    return retVal

def generatePy(datacls):

    devid = getNextId()
    _, _, _ = datacls.get_training_data(devid)
    _, _, _ = datacls.get_testing_data(devid)

    tensor_train_x, tensor_train_y, rot = datacls.get_training_data(devid)
    train_y = tensor_train_y.numpy()

    train_x = tensor_train_x.numpy()

    counts = np.bincount(train_y)
    target = str(np.argmax(counts))

    #histInput = list(map(str, train_y.tolist()))
    #return devid, target, rot, HistSummary(histInput)

    histInput = {}
    histMatInput = {}

    labelSpace = list(map(str, np.unique(train_y)))
    for label in labelSpace:
        histInput[label] = []

    for yIdx in range(len(train_y)):
        label = str(train_y[yIdx])
        xarr = train_x[yIdx,:].flatten()
        counts, xLabels = np.histogram(xarr, bins=100, range=(0,1))
        sd = []
        for xIdx, numericLabel in enumerate(xLabels[:-1]):
            count = counts[xIdx]
            xLab = "b" + str(numericLabel)
            sd = sd + count*[xLab]

        histInput[label] += sd

    for label in labelSpace:
        hip = histInput[label]
        histMatInput[label] = HistSummary(hip)

    return devid, target, rot, HistMatSummary(histMatInput)

def genDevices(count, datacls, eps=1.0):

    devs = {}
    labels = set()

    for i in range(count):

        devid, target, rot, hs = generatePy(datacls)
        devs[devid] = [target, hs, rot]

        labels = labels.union(hs.getKeys())
        datacls.clear()

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

def predictClusters(devs):

    n_available_devices = len(devs)
    xKeyspace = set([])
    yKeyspace = set([])
    summaries = []

    # We need a deterministic ordering of keys
    dev_keys = copy.deepcopy(list(devs.keys()))

    for devId in dev_keys:
        histSummary = devs[devId][1]
        summaries.append(histSummary)
        yKeys = histSummary.getKeys()
        yKeyspace = yKeyspace.union(yKeys)
        for key in yKeys:
            xKeyspace = xKeyspace.union(histSummary.at(key).getKeys())
            
    dev_clusters = cluster_mat(summaries, list(xKeyspace), list(yKeyspace))
    return dev_clusters

#def predictClusters(devs, labels):

#    hlist = []
#    for devid in devs.keys():
#        hlist.append(devs[devid][1])

#    classes = cluster_hist(hlist, labels)
#    return classes

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

from mnist_rot_own import MNIST_ROT_OWN
datacls = MNIST_ROT_OWN()

print("Downloading Data...",end="")
sys.stdout.flush()
datacls.download_data()

print(" Done.")

labels, devs = genDevices(10, datacls)
pred = predictClusters(devs)

#addNoise(devs, 1.0)
#addNoise(devs, 0.001)
for idx, k in enumerate(devs.keys()):
    print("Dev",k,":")
    print("  Majority Label:    ", devs[k][0])
    print("  Rotation:          ", devs[k][2])
    print("  Cluster Assignment:", pred[idx])
    #print("  ", devs[k][1].toJson())
    #print("  ", devs[k][2])

#clust = getTrueClusters(devs, labels)
#pred = predictClusters(devs, labels)
#print(clust)
#print(pred)

#acc = computeAcc(devs, clust, pred)
#print("Accuracy: " + str(acc * 100) + "%")

