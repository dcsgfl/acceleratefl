import os
import gzip
import shutil
import numpy as np

import urllib.parse

from urllib.request import urlretrieve
from data import data

######constants######
# Absolute path of "data" directory 
DATADIR = os.path.join(os.path.abspath(os.path.dirname(__file__)))

class mnist(data):
    def __init__(self) -> None:
        super().__init__()
        self.path = os.path.join(DATADIR, 'mnist')
        self.url = 'http://yann.lecun.com/exdb/mnist/'
        self.gz = [ 'train-images-idx3-ubyte.gz',
                    't10k-images-idx3-ubyte.gz',
                    'train-labels-idx1-ubyte.gz',
                    't10k-labels-idx1-ubyte.gz']
                        
    def download_data(self):
        # Create mnist directory is non-existent
        os.makedirs(self.path, exist_ok=True)

        # Download mnist gz files if non-existent
        for gz in self.gz:
            if gz not in os.listdir(self.path):
                urlretrieve(urllib.parse.urljoin(self.url, gz), os.path.join(self.path, gz))
                print('Downloaded ', gz, ' from ', self.url)
        
        # Retrieve train and test data from gz files
        for gz in self.gz:
            with gzip.open(os.path.join(self.path, gz), 'rb') as gzobj:
                if gz == 'train-images-idx3-ubyte.gz':
                    # skip first 16 bytes (magicno(4), image count(4), row count(4) = 28, column count(4) = 28)
                    self.train_x = np.frombuffer(gzobj.read(), np.uint8, offset=16).reshape(-1, 28, 28)
                
                if gz == 't10k-images-idx3-ubyte.gz':
                    # skip first 16 bytes (magicno(4), image count(4), row count(4) = 28, column count(4) = 28)
                    self.test_x = np.frombuffer(gzobj.read(), np.uint8, offset=16).reshape(-1, 28, 28)
                
                if gz =='train-labels-idx1-ubyte.gz':
                    # skip first 8 bytes (magicno(4), label count (4))
                    self.train_y = np.frombuffer(gzobj.read(), np.uint8, offset=8)

                if gz =='t10k-labels-idx1-ubyte.gz':
                    # skip first 8 bytes (magicno(4), label count (4))
                    self.test_y = np.frombuffer(gzobj.read(), np.uint8, offset=8)

        # remove the mnist directory after use
        shutil.rmtree(self.path)

    def get_training_data(self):
        raise NotImplementedError("ERROR: get_training_data unimplemented")
    
    def get_testing_data(self):
        raise NotImplementedError("ERROR: get_testing_data unimplemented")


if __name__ == '__main__':
    cls = mnist()

    cls.download_data()
