import os
import sys
import argparse
import torch

import syft as sy
from syft.workers import websocket_server
from torchvision import transforms

# Add data folder to path
pwd = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'data')
sys.path.append(pwd)

def start_websocker_server_worker(id, host, port, dataset, hook, verbose):
    server = websocket_server.WebsocketServerWorker(
        id = id,
        host = host,
        port = port,
        hook = hook,
        verbose = verbose)

    # Dataset class selection
    from datasetFactory import DatasetFactory as dftry
    datacls = dftry.getDataset(dataset)

    # Training data
    train_data, train_targets = datacls.get_training_data()
    dataset_train = sy.BaseDataset(
        data = train_data,
        targets = train_targets,
        transform = transforms.Compose([transforms.ToTensor()])
    )
    server.add_dataset(dataset_train, key = dataset + '_TRAIN')

    # Testing data
    test_data, test_targets = datacls.get_testing_data()
    dataset_test = sy.BaseDataset(
        data = test_data,
        targets = test_targets,
        transform = transforms.Compose([transforms.ToTensor()])
    )
    server.add_dataset(dataset_test, key = dataset + '_TEST')

    server.start()

    return server

if __name__ == '__main__':

    #Parse arguments
    parser = argparse.ArgumentParser(description='Run websocket server worker')

    parser.add_argument(
        '--port',
        '-p',
        type=int,
        help='port number on which websocket server will listen: --port 8777',
    )

    parser.add_argument(
        '--host',
        '-h',
        default='localhost',
        help='host on which the websocket server worker should be run: --host 1.2.3.4',
    )

    parser.add_argument(
        '--id',
        type=str,
        help='name of the websocket server worker: --id alice'
    )

    parser.add_argument(
        '--dataset',
        '-ds',
        type=str,
        help='dataset used for the model: --dataset CIFAR10'
    )

    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='start websocket server worker in verboase mode: --verbose'
    )

    args = parser.parse_args()

    # Hook PyTorch to add extra functionalities to support FL
    hook = sy.TorchHook(torch)

    server = start_websocker_server_worker(
        id = args.id,
        host = args.host,
        port = args.port,
        dataset = args.dataset,
        hook = hook,
        verbose = args.verbose
    )

