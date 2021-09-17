"""
tiflScheduler.py

Scheduler based on the TiFL paper
UMN DCSG, 2021
"""

import random

from scheduler import Scheduler

class TIFLSched(Scheduler):

    def __init__(self):
        pass

    def select_worker_instances(self, available_devices, client_threshold):
        selected_devices = {}
        device_keys = random.sample(list(available_devices.keys()), client_threshold)
        for key in device_keys:
            selected_devices[key] = available_devices[key].copy()
        return(selected_devices)
    
    def notify_worker_update(self, all_devices):
        pass

    class Factory:
        def get(self):
            return TIFLSched()
