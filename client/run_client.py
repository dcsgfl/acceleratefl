import os
import sys
import argparse
import torch
import grpc
import psutil

import syft as sy
from syft.workers import websocket_server
from torchvision import transforms

import threading
import time

import logging

# Add data folder to path
pwd = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'data')
sys.path.append(pwd)

# Add common folder to path
pwd = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'common')
sys.path.append(pwd)

from utilities import Utility as util
import devicetocentral_pb2
import devicetocentral_pb2_grpc

devid = ""
def register_to_central(args):
    with grpc.insecure_channel(args.centralip + ':50051') as channel:
        stub = devicetocentral_pb2_grpc.DeviceToCentralStub(channel)
        logging.info('Registering to central server: ' + args.centralip + ':50051')
        resp = stub.RegisterToCentral(
            devicetocentral_pb2.DeviceInfo (
                ip = args.host,
                flport = args.port
            )
        )

        logging.info('Registration complete')

        if resp.success :
            logging.info(args.host + ':' + str(args.port) + ' registered with id' + resp.id + '...')
            global devid
            devid = resp.id
            return True

    return False

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

def parse_arguments(args = sys.argv[1:]):
    parser = argparse.ArgumentParser(description='Run websocket server worker')

    parser.add_argument(
        '--port',
        default = util.get_free_port(),
        type=int,
        help='port number on which websocket server will listen: --port 8777',
    )

    parser.add_argument(
        '--host',
        type=str,
        default='localhost',
        help='host on which the websocket server worker should be run: --host 1.2.3.4',
    )

    parser.add_argument(
        '--id',
        type=str,
        default='alice',
        help='name of the websocket server worker: --id alice'
    )

    parser.add_argument(
        '--dataset',
        '-ds',
        type=str,
        default='CIFAR10',
        help='dataset used for the model: --dataset CIFAR10'
    )

    parser.add_argument(
        '--centralip',
        '-cip',
        type=str,
        default='localhost',
        help = 'central server ip address: --centralip 1.2.3.4'
    )

    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='start websocket server worker in verbose mode: --verbose'
    )

    args = parser.parse_args(args = args)
    return args

def heartbeat(args):

    while(True):
        time.sleep(5)
        load = psutil.os.getloadavg()
        virt_mem = psutil.virtual_memory()
        battery = psutil.sensors_battery()

        with grpc.insecure_channel(args.centralip + ':50051') as channel:
            stub = devicetocentral_pb2_grpc.DeviceToCentralStub(channel)
            logging.info('Heat beat to server...')
            resp = stub.HeartBeat(
                devicetocentral_pb2.Ping (
                    cpu_usage = psutil.cpu_percent(),
                    ncpus = psutil.cpu_count(),
                    load15 = load[2],
                    virtual_mem = virt_mem.available/(1024*1024*1024),
                    battery = battery.percent,
                    id = devid
                )
            )

            if resp.ack :
                logging.info('Heart beat success...')
            else:
                logging.info('Connection to server failed...')
                return

if __name__ == '__main__':

    #Parse arguments
    args = parse_arguments()

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S')

    # grpc call to central server to register
    stat = register_to_central(args)
    if not stat:
        print('Registration to central failed...')
        sys.exit()
    
    heartbeat_service = threading.Thread(target=heartbeat, args=(args, ))
    heartbeat_service.start()
    heartbeat_service.join()

    # Hook PyTorch to add extra functionalities to support FL
    # hook = sy.TorchHook(torch)

    # server = start_websocker_server_worker(
    #     id = args.id,
    #     host = args.host,
    #     port = args.port,
    #     dataset = args.dataset,
    #     hook = hook,
    #     verbose = args.verbose
    # )