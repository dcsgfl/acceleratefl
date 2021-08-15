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
    return [0.25, 0.25, 0.25, 0.25] # todo

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
    dev["cpu_usage"] = randrange(100)

    return dev

keySpace = ['0','1','2','3']

devs = {}
devs["dev_0"] = getDevice("dev_0", 1000, keySpace)
devs["dev_1"] = getDevice("dev_1", 1000, keySpace)
devs["dev_2"] = getDevice("dev_2", 1000, keySpace)
devs["dev_3"] = getDevice("dev_3", 1000, keySpace)

#print(devs)
py = PYSched()

ignored_thresh = 0
selected_devs = py.select_worker_instances(devs, ignored_thresh)
print(selected_devs)
