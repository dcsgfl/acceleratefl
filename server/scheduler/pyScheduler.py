import os
import sys
import numpy as np

pwd = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..', 'common','summary')
sys.path.append(pwd)

from cluster import cluster_hist
from hist import HistSummary

class PYSched:
    def __init__(self):
        pass

    def select_worker_instances(self, available_devices, client_threshold):

        n_available_devices = len(available_devices)
        summaries = []
        keyspace = set([])

        # unpack histogram infor and create a list
        for device in available_devices:
            histSummary = HistSummary()
            histSummary.fromJson(device['summary'])
            summaries.append(histSummary)
            keyspace.union(histSummary.getKeys())
        
        # identify cluster of each device
        dev_clusters = cluster_hist(summaries, list(keyspace))
        for idx, device in enumerate(available_devices):
            device['cluster'] = dev_clusters[idx]

        return available_devices
        #available_devices[request.id]['cpu_usage']
        #available_devices[request.id]['ncpus']
        #available_devices[request.id]['load']
        #available_devices[request.id]['virtual_mem']
        #available_devices[request.id]['battery']


        #     if(dev.id in selected_devid):
        #         res.append(dev)
        # return (res)
    
    class Factory:
        def get(self):
            return PYSched()
