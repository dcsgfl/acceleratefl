"""
sched_test.py

Test cases for schedulers
UMN DCSG, 2021
"""

import os
import sys

pwd = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'server','scheduler')
sys.path.append(pwd)

pwd = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'common','summary')
sys.path.append(pwd)

import numpy as np
from random import randrange

from hist import HistSummary
from pyScheduler import PYSched

def getRandomConvexCombination(dim):
    probs = [0.1, 0.1, 0.1, 0.1]
    ridx = randrange(4)
    probs[ridx] = 0.7
    return probs

def getDevice(dev_id, numDataPoints, keySpaceList):

    dev = {}

    probs = getRandomConvexCombination(len(keySpaceList))
    label_counts = np.random.multinomial(numDataPoints, probs)
    labels = []
    for idx, key in enumerate(keySpaceList):
        labels = labels + [key] * label_counts[idx]

    histSummary = HistSummary(list(map(str, labels)))

    dev["id"] = dev_id
    dev["summary"] = histSummary.toJson()
    dev["cpu_usage"] = float(randrange(100)) / 100.0
    dev["loss"] = float(randrange(100)) / 100.0
    #dev["loss"] = 1.0

    return dev

keySpace = ['0','1','2','3']

devs = {}
devs["dev_0"]  = getDevice("dev_0", 1000, keySpace)
devs["dev_1"]  = getDevice("dev_1", 1000, keySpace)
devs["dev_2"]  = getDevice("dev_2", 1000, keySpace)
devs["dev_3"]  = getDevice("dev_3", 1000, keySpace)
devs["dev_4"]  = getDevice("dev_4", 1000, keySpace)
devs["dev_5"]  = getDevice("dev_5", 1000, keySpace)
devs["dev_6"]  = getDevice("dev_6", 1000, keySpace)
devs["dev_7"]  = getDevice("dev_7", 1000, keySpace)
devs["dev_8"]  = getDevice("dev_8", 1000, keySpace)
devs["dev_9"]  = getDevice("dev_9", 1000, keySpace)
devs["dev_10"] = getDevice("dev_10", 1000, keySpace)
devs["dev_11"] = getDevice("dev_11", 1000, keySpace)

py = PYSched()
py.notify_worker_update(devs)

print("PY TEST")
for k in devs.keys():
    print(k,":",devs[k])

thresh = 4
selected_devs = py.select_worker_instances(devs, thresh)
print("SELECTED")
for k in selected_devs.keys():
    print(k,":",selected_devs[k])

#print("TIFL TEST")
#tifl = TIFLSched()
#tifl.notify_worker_update(devs)
#for key in devs.keys():
#    print(devs[key])

#print()
#ignored_thresh = 2
#selected_devs = tifl.select_worker_instances(devs, ignored_thresh)
#print(selected_devs)
