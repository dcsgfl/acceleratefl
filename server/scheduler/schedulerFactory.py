

class SchedulerFactory:
    factories = {}
    
    def addFactory(id, dftory):
        SchedulerFactory.factories.put[id] = dftory
    addFactory = staticmethod(addFactory)
    
    def getModel(id):
        if id not in SchedulerFactory.factories:
            SchedulerFactory.factories[id] = eval(id + '.Factory()')

        return SchedulerFactory.factories[id].get()
    getDataset = staticmethod(getModel)