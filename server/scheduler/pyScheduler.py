import os
import sys
import numpy as np

pwd = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..', 'common','summary')
sys.path.append(pwd)

from cluster import cluster_hist

class PYSched:
    def __init__(self):
        pass

    def select_worker_instances(self, available_devices, client_threshold):
        n_available_devices = len(available_devices)
        device_summaries = [[]] * n_available_devices
        iter = 0
        keyspace = []
        for device in available_devices:
            device_summaries[iter] = device['summary'].copy()
            iter = iter + 1
            if iter == n_available_devices - 1:
                keyspace = device['labels']
            
        dev_clusters = cluster_hist(device_summaries, keyspace)


        #     if(dev.id in selected_devid):
        #         res.append(dev)
        # return (res)
    
    class Factory:
        def get(self):
            return PYSched()