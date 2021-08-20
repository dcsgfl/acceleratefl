import math
import numpy as np

from scheduler import Scheduler

class PXYSched(Scheduler):

    def __init__(self):
        self.G_CACHED_SUMMARIES = {}

    def hellingerDist(self, hist_p, hist_q):

        dp = np.sum(hist_p)
        dq = np.sum(hist_q)

        dist = 0.0
        for i in range(len(hist_p)):
            prob_p = hist_p[i] / dp
            prob_q = hist_q[i] / dq

            dist += (math.sqrt(prob_p) - math.sqrt(prob_q))**2


        return (1/math.sqrt(2)) * math.sqrt(dist)

    #
    # Takes local client data as input and produces a lower-dimensional
    # point esimate for the conditional distribution of X given Y.
    # Expected output shape: (# classes, 256)
    # (for each class, count of images, histogram)
    #
    # THIS ASSUMES deviceTargets contains values in [0, nClasses - 1]
    #
    def getClientDataSummary(self, deviceId, deviceData, deviceTargets, nClasses):
    
        global G_CACHED_SUMMARIES

        if deviceId in self.G_CACHED_SUMMARIES.keys():
            return {deviceId: self.G_CACHED_SUMMARIES[deviceId]}

        summary = np.zeros((nClasses, 1 + 256))   # First cell is the count

        for row in range(deviceData.shape[0]):

            curClass = deviceTargets[row]
            if curClass >= nClasses or curClass < 0:
                print("ERR getClientDataSummary(): Expected class labels to be between [0, nClasses -1]")
                sys.exit(1)

            summary[curClass][0] += 1
            values, counts = np.unique(deviceData[row], return_counts=True)

            for idx, value in enumerate(values):
                summary[curClass][1 + value] += counts[idx]

        self.G_CACHED_SUMMARIES[deviceId] = summary
        return {deviceId: summary}

    #
    # Takes all summaries collected from client devices and produces a
    # pairwise comparision of the conditional distributions.
    #
    # Input Shape: (# devices, (output shape of getClientDataSummary()))
    # Ouput Shape: (# devices, # devices - 1)
    #
    def compareClientDevices(self, devIds, allDeviceSummaries, nClasses):

        nDevices = allDeviceSummaries.shape[0]
        compMat = {}

        for i in range(nDevices):

            tmpDict = {}

            for j in range(nDevices):

                if i == j:
                    continue

                dev_i = allDeviceSummaries[i]
                dev_j = allDeviceSummaries[j]

                totalDist = 0
                for k in range(nClasses):
                    if dev_i[k][0] > 0 and dev_j[k][0] > 0:
                        totalDist += self.hellingerDist(dev_i[k][1:], dev_j[k][1:])
                    elif dev_i[k][0] > 0 or dev_j[k][0] > 0:
                        totalDist += 1

                tmpDict[devIds[j]] = totalDist

            sortedDict = dict(sorted(tmpDict.items(), key=lambda item: item[1]))
            compMat[devIds[i]] = sortedDict

        return compMat

    def select_worker_instances(self, _epoch, client_list, num_of_selected, dev_profile):
        if(_epoch==1):
            return(client_list)

        # from realmnistds import REALMNISTDS
        # cls = REALMNISTDS()

        from cifar10 import CIFAR10
        cls = CIFAR10()
        
        summaries = []
        devIds = []
        nClasses = 10   # todo, can we get this dynamically?

        # Get summaries from each device
        for dev in client_list:
            print(_epoch, dev.id)
            dev_data, dev_targets = cls.get_training_part(dev.id)
            summary = self.getClientDataSummary(dev.id, dev_data, dev_targets, nClasses)
            devIds.append(dev.id)
            summaries.append(summary[dev.id])

        # Get comparison matrix for best matches
        dev_summaries = np.array(summaries)
        compMatrix = self.compareClientDevices(devIds, dev_summaries, nClasses)

        selected_devid=set(compMatrix.keys())
        for kk in compMatrix.keys():
            if(len(compMatrix[kk])>0):
                ll=list(compMatrix[kk].keys())
                # print(kk, ll)
                dev1=dev_profile[ll[0]]
                dev0=dev_profile[kk]
                if(dev1<dev0):
                    selected_devid.remove(kk)
        res=[]
        for dev in client_list:
            if(dev.id in selected_devid):
                res.append(dev)
        return (res)

    def notify_worker_update(self, all_devices):
        pass

    class Factory:
        def get(self):
            return PXYSched()
