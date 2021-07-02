import os
import gzip
import shutil
import numpy as np

import urllib.parse

from urllib.request import urlretrieve
from DatasetFactory import Dataset

######constants######
# Absolute path of "data" directory 
DATADIR = os.path.join(os.path.abspath(os.path.dirname(__file__)))

class flickr(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.path = os.path.join(DATADIR, 'flickr')
        self.url = 'https://zenodo.org/record/3676081/files'
        self.gz = 'geo_animal.tar.gz'
                        
    def download_data(self):
        # Create flickr directory is non-existent
        os.makedirs(self.path, exist_ok=True)

        # Download flickr gz file if non-existent
        if self.gz not in os.listdir(self.path):
            urlretrieve(urllib.parse.urljoin(self.url, self.gz), os.path.join(self.path, self.gz))
            print('Downloaded ', self.gz, ' from ', self.url)

        # TO DO - download images and convert to bin files depending on the location.

    def get_training_data(self):
        raise NotImplementedError("ERROR: get_training_data unimplemented")
    
    def get_testing_data(self):
        raise NotImplementedError("ERROR: get_testing_data unimplemented")


if __name__ == '__main__':
    cls = flickr()

    cls.download_data()
