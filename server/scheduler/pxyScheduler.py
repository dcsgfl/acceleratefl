import os
import sys
import numpy as np
import copy
from collections import Counter

pwd = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..', 'common','summary')
sys.path.append(pwd)

from cluster import cluster_mat
from hist import HistSummary, HistMatSummary

from scheduler import Scheduler

import logging

class PXYSched(Scheduler):

    def __init__(self):
        self.cluster_info = None

    def do_clustering(self, all_devices):

        n_available_devices = len(all_devices)
        summaries = []
        xKeyspace = set([])
        yKeyspace = set([])

        # We need a deterministic ordering of keys
        dev_keys = copy.deepcopy(list(all_devices.keys()))

        for devId in dev_keys:
            histSummary = HistMatSummary()
            histSummary.fromJson(all_devices[devId]['summary'])
            summaries.append(histSummary)
            yKeys = histSummary.getKeys()
            yKeyspace = yKeyspace.union(yKeys)
            for key in yKeys:
                xKeyspace = xKeyspace.union(histSummary.at(key).getKeys())
            
        dev_clusters = cluster_mat(summaries, list(xKeyspace), list(yKeyspace))
        nextClustId = max(dev_clusters) + 1
        self.cluster_info = {}

        for idx, devId in enumerate(dev_keys):

            clustId = dev_clusters[idx]

            # Assign the -1 values to their own cluster
            if clustId == -1:
                clustId = nextClustId
                nextClustId += 1

            if clustId not in self.cluster_info.keys():
                self.cluster_info[clustId] = {}
                self.cluster_info[clustId]["count"] = 0

            self.cluster_info[clustId]["count"] += 1

            all_devices[devId]['cluster'] = clustId
            #logging.info("cpu_usage")
            #logging.info(all_devices[devId]['cpu_usage'])

        for clustId in self.cluster_info.keys():
            num = float(self.cluster_info[clustId]["count"])
            self.cluster_info[clustId]["prop"] = num / float(len(dev_keys))

        logging.info("CLUSTERS ASSIGNED: " + str(Counter(dev_clusters)))


    def select_worker_instances(self, available_devices, client_threshold):

        if self.cluster_info is None:
            # Someone forgot to call notify_worker_update() ...
            # TODO this is really a bug... We might not have all the devices
            self.do_clustering(available_devices)

        return self._schedule_clusters(self.cluster_info, available_devices, client_threshold)
        # This is a little slow... (n^2) but just moving on for now
        #selected_devices = {}
        #for clusterId in self.cluster_ids: # For each identified cluster

        #    bestDev = {}
        #    for devId in available_devices.keys():  # For each device in that cluster

        #        curDev = available_devices[devId].copy()
    
        #        if curDev['cluster'] == clusterId:

        #            if len(bestDev.keys()) == 0:
                        # This is the first device in cluster
        #                bestDev = curDev.copy()
        #            elif curDev["cpu_usage"] < bestDev["cpu_usage"]:
                        # Is this device any faster?
        #                bestDev = curDev.copy()

                # TODO: these keys are also available to us...
                #available_devices[request.id]['cpu_usage']
                #available_devices[request.id]['ncpus']
                #available_devices[request.id]['load']
                #available_devices[request.id]['virtual_mem']
                #available_devices[request.id]['battery']

        #    selected_devices[bestDev["id"]] = bestDev.copy()

        #return selected_devices

    def notify_worker_update(self, all_devices):
        self.do_clustering(all_devices)

    class Factory:
        def get(self):
            return PXYSched()
