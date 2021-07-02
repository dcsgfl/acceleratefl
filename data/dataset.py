class Dataset:
    def __init__(self) -> None:
        pass
        # raise NotImplementedError("ERROR: __init__ unimplemented")

    def download_data(self):
        raise NotImplementedError("ERROR: download_data unimplemented")
    
    def get_training_data(self):
        raise NotImplementedError("ERROR: get_training_data unimplemented")
    
    def get_testing_data(self):
        raise NotImplementedError("ERROR: get_testing_data unimplemented")