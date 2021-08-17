import os
import sys
import numpy as np
import copy

pwd = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..', 'common','summary')
sys.path.append(pwd)

from cluster import cluster_hist
from hist import HistSummary

class PYSched:
    def __init__(self):
        pass

    # TODO: seems like we should have a cached version of the clusters somewhere
    #       so we are not doing this every epoch...
    def select_worker_instances(self, available_devices, client_threshold):

        n_available_devices = len(available_devices)
        summaries = []
        keyspace = set([])

        # We need a deterministic ordering of keys
        dev_keys = copy.deepcopy(list(available_devices.keys()))

        for devId in dev_keys:
            device = available_devices[devId]
            histSummary = HistSummary()
            histSummary.fromJson(device['summary'])
            summaries.append(histSummary)
            keyspace = keyspace.union(histSummary.getKeys())
            
        dev_clusters = cluster_hist(summaries, list(keyspace))

        for idx, devId in enumerate(dev_keys):
            available_devices[devId]['cluster'] = dev_clusters[idx]

        # This is a little slow... (n^2) but just moving on for now
        selected_devices = {}
        for clusterId in set(dev_clusters): # For each identified cluster

            bestDev = {}
            for devId in dev_keys:          # For each device in that cluster

                curDev = available_devices[devId].copy()

                if curDev['cluster'] == clusterId:

                    if len(bestDev.keys()) == 0:
                        # This is the first device in cluster
                        bestDev = curDev.copy()
                    elif curDev["cpu_usage"] < bestDev["cpu_usage"]:
                        # Is this device any faster?
                        bestDev = curDev.copy()

                # TODO: these keys are also available to us...
                #available_devices[request.id]['cpu_usage']
                #available_devices[request.id]['ncpus']
                #available_devices[request.id]['load']
                #available_devices[request.id]['virtual_mem']
                #available_devices[request.id]['battery']

            selected_devices[bestDev["id"]] = bestDev.copy()

        return selected_devices

    class Factory:
        def get(self):
            return PYSched()
