"""
tiflScheduler.py

Scheduler based on the TiFL paper
UMN DCSG, 2021
"""

import copy
import random
import logging

from scheduler import Scheduler

class TIFLSched(Scheduler):

    def __init__(self):
        self.n_tiers = 0

    def select_worker_instances(self, available_devices, client_threshold):
        selected_devices = {}
        device_keys = random.sample(list(available_devices.keys()), client_threshold)
        for key in device_keys:
            selected_devices[key] = available_devices[key].copy()
        return(selected_devices)
    
    def notify_worker_update(self, all_devices):

        n_available_devices = len(all_devices)
        self.n_tiers = min(n_available_devices, 5)

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

            for i in range(ub):
                all_devices[s[assignedCount]]['tifl_tier'] = sid
                assignedCount += 1

    class Factory:
        def get(self):
            return TIFLSched()

