"""
scheduler.py

Parent scheduler class for flsys
UMN DCSG, 2021
"""

import math
from random import choices

import numpy as np

"""
All schedulers should implement the following interfaces
"""
class Scheduler:

    def __init__(self):
        pass

    """
    Called by the server when a worker is added or data distributions change

    @param all_devices  A list of all devices the server is aware of
    """
    def notify_worker_update(self, all_devices):
        raise NotImplementedError

    """
    Called by the server each epoch when devices must be selected for training

    @param available_devices    The devices available for training this epoch
    @param client_threshold     Max number of devices allowed for training
    """
    def select_worker_instances(self, available_devices, client_threshold):
        raise NotImplementedError

    """
    --- NOT ABSTRACT ---
    Called by child implementations to perform scheduling common to the
    techniques that perform clustering.

    * Assumes there are cluster ids assigned to available_devices
    * Assumes self.cluster_ids contains the set of valid ids
    """
    def _schedule_clusters(self, clustInfo, available_devices, client_threshold):

        closs = {}
        clat  = {}
        cdevs = {}

        RHO = 0.1   # Weight loss and latency reduction equally
        USE_QUASI_RANDOM = False

        #
        # Compute the total loss in each cluster
        # and max latency
        #
        for devId in available_devices:

            dev = available_devices[devId]
            clustId = dev["cluster"]
            #sqloss = math.pow(dev['loss'], 2.0)
            sqloss = dev['loss']
            latency = dev['cpu_usage']

            if clustId not in closs.keys():
                closs[clustId] = sqloss
                clat[clustId]  = latency
                cdevs[clustId] = []
            else:
                closs[clustId] += sqloss
                clat[clustId]  += float(latency)

            cdevs[clustId].append(dev.copy())

        #
        # Compute the averages and max latency
        #
        maxLatency = 0.0
        totalLoss = 0.0
        for cid in closs.keys():
            closs[cid] /= float(clustInfo[cid]['count'])
            clat[cid]  /= float(clustInfo[cid]['count'])
            maxLatency = max(clat[cid], maxLatency)
            totalLoss += closs[cid]

        #
        # Compute sampling weights
        #
        clusters = []
        probs = []
        for cid in closs.keys():
            latRed = 1.0 - (clat[cid] / maxLatency)
            normLoss = closs[cid] / totalLoss
            p = RHO*latRed + (1.0 - RHO)*normLoss

            #print("cluster",cid,"latRed",round(latRed,6),"loss",round(normLoss,6),"weight",round(p,6))

            clusters.append(cid)
            probs.append(p)

        #
        # Sort the devices in each cluster by their utility
        #
        for cid in clusters:

            utility = []
            for dev in cdevs[cid]:
                #util = dev['loss'] / (1.0 + (dev['cpu_usage'] / 100.0))
                util = 1.0 - dev['cpu_usage']
                utility.append(util)

            s = [x for _, x in sorted(zip(utility, cdevs[cid]),
                                      key=lambda pair: pair[0],
                                      reverse=True)]
            cdevs[cid] = s

        #
        # Sample clusters and select the best devices
        #
        selected = {}
        selectedClusters = []
        normProbs = np.array(probs) / float(sum(probs))

        if USE_QUASI_RANDOM:

            # OK let's take the guesswork out of this...
            # time for some quasi-random draws
            for idx, cid in enumerate(clusters):

                expected = int(normProbs[idx] * client_threshold)

                devs = cdevs[cid]
                ccnt = min(expected, len(devs))

                for i in range(ccnt):
                    dev = devs[0]
                    selected[dev['id']] = dev
                    selectedClusters.append(str(cid))
                    del devs[0]

                if len(devs) == 0:
                    # Remove this cluster from consideration
                    # del clusters[i]
                    # del probs[i]
                    probs[idx] = 0.0

            while len(selectedClusters) < client_threshold:
                clust = choices(clusters, weights=probs)[0]
                devs = cdevs[clust]

                dev = devs[0]
                selected[dev['id']] = dev
                selectedClusters.append(str(clust))
                del devs[0]

                if len(devs) == 0:
                    # Remove this cluster from consideration
                    # del clusters[i]
                    # del probs[i]
                    cidx = clusters.index(clust)
                    probs[cidx] = 0.0

        else:

            count = 0
            while count < client_threshold:

                idx = choices(range(len(clusters)), weights=probs)[0]
                clust = clusters[idx]
                devs = cdevs[clust]
                dev = devs[0]    # this will just work if sorted jsw

                if dev['loss'] > 0.0:
                    selected[dev['id']] = dev
                    selectedClusters.append(str(clust))
                    count += 1

                del devs[0]
                if len(devs) == 0:
                    # Remove this cluster from consideration
                    # del clusters[idx]
                    # del probs[idx]
                    probs[idx] = 0.0

        #print("Scheduled clusters: ", ' '.join(map(str, selectedClusters)))
        return selected

