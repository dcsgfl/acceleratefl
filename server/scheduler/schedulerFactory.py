from rndScheduler import RNDSched
from pyScheduler import PYSched
from pxyScheduler import PXYSched
from tiflScheduler import TIFLSched
from oortScheduler import OORTSched

class SchedulerFactory:
    factories = {}
    
    def addFactory(id, dftory):
        SchedulerFactory.factories.put[id] = dftory
    addFactory = staticmethod(addFactory)
    
    def getScheduler(id):
        if id not in SchedulerFactory.factories:
            SchedulerFactory.factories[id] = eval(id + '.Factory()')

        return SchedulerFactory.factories[id].get()
    getDataset = staticmethod(getScheduler)