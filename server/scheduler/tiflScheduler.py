"""
tiflScheduler.py

Scheduler based on the TiFL paper
UMN DCSG, 2021
"""

import copy
import random
import logging

import numpy as np

from scheduler import Scheduler
from random import choices

class TIFLSched(Scheduler):

    def __init__(self):
        self.n_tiers = 0
        self.tier_probs = None
        self.tier_credits = None
        self.tier_counts = None

    def select_worker_instances(self, available_devices, client_threshold):

        device_keys = list(available_devices.keys())
        selected_devices = {}

        # Just update probabilities every round...
        self.tier_probs = np.zeros(self.n_tiers)

        for devKey in device_keys:
            dev = available_devices[devKey]
            tierIdx = int(dev['tifl_tier']) - 1
            self.tier_probs[tierIdx] += dev['loss']

        for tierIdx in range(self.n_tiers):
            self.tier_probs[tierIdx] /= float(self.tier_counts[tierIdx])

        # OK, now, select a tier
        idx = -1
        weights = np.copy(self.tier_probs)
        while True:

            idx = choices(range(self.n_tiers), weights=weights)[0]
            if self.tier_credits[idx] > 0:
                break

            # Remove this option
            weights[idx] = 0.0

        numDevs = int(min(self.tier_counts[idx], client_threshold))
        self.tier_credits[idx] -= 1
        tierStr = str(idx + 1)
        keys = []

        for devKey in device_keys:
            dev = available_devices[devKey]
            if dev['tifl_tier'] == tierStr:
                keys.append(devKey)

        samp = random.sample(keys, numDevs)
        for key in samp:
            selected_devices[key] = available_devices[key].copy()

        return(selected_devices)

    def notify_worker_update(self, all_devices):

        n_available_devices = len(all_devices)

        self.n_tiers = min(n_available_devices, 5)
        self.tier_probs = np.ones(self.n_tiers) / float(self.n_tiers)
        self.tier_credits = np.array([80, 50, 30, 20, 20])
        self.tier_counts = np.zeros(self.n_tiers)

        devsPerTier = int(n_available_devices / int(self.n_tiers))

        # We need a deterministic ordering of keys
        dev_keys = copy.deepcopy(list(all_devices.keys()))

        utility = []
        for key in dev_keys:
            dev = all_devices[key]
            util = 1.0 - dev['cpu_usage']
            utility.append(util)

        s = [x for _, x in sorted(zip(utility, dev_keys),
                                  key=lambda pair: pair[0],
                                  reverse=True)]

        assignedCount = 0
        for tid in range(self.n_tiers):
            sid = str(tid + 1)

            ub = devsPerTier
            if tid == (self.n_tiers - 1):
                ub = n_available_devices - assignedCount

            self.tier_counts[tid] = ub

            for i in range(ub):
                all_devices[s[assignedCount]]['tifl_tier'] = sid
                assignedCount += 1

    class Factory:
        def get(self):
            return TIFLSched()

