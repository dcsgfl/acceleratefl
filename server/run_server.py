# The model owner is considered as client
import os
import sys
import time
import asyncio
import argparse
import threading
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import syft as sy
from syft.workers import websocket_client
from syft.workers.websocket_client import WebsocketClientWorker

from concurrent import futures
import grpc
import logging

# Add common folder to path
pwd = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'common')
sys.path.append(pwd)

# Add models folder path
pwd = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'models')
sys.path.append(pwd)

# Add scheduler folder to path
pwd = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'scheduler')
sys.path.append(pwd)

import devicetocentral_pb2
import devicetocentral_pb2_grpc

from utilities import Utility as util
from modelFactory import ModelFactory as mdlftry
from schedulerFactory import SchedulerFactory as schedftry

class DeviceToCentralServicer(devicetocentral_pb2_grpc.DeviceToCentralServicer):
    
    def __init__(self):
        super().__init__()
        self.available_devices = {}
        self.lock = threading.Lock()
        
    def RegisterToCentral(self, request, context):
        msg = request.ip + ':' + str(request.flport)
        id = util.get_id(msg)
        
        self.lock.acquire()
        if id not in self.available_devices:
            self.available_devices[id] = {}
            self.available_devices[id]['id'] = id
            self.available_devices[id]['ip'] = request.ip
            self.available_devices[id]['flport'] = request.flport
        self.lock.release()

        logging.info('Registered client ' + request.ip + ':' + str(request.flport) + '...')
        
        return devicetocentral_pb2.RegStatus(
            success = True,
            id = id
        )
    
    def HeartBeat(self, request, context):
        self.lock.acquire()
        self.available_devices[request.id]['cpu_usage'] = request.cpu_usage
        self.available_devices[request.id]['ncpus'] = request.ncpus
        self.available_devices[request.id]['load'] = request.load15
        self.available_devices[request.id]['virtual_mem'] = request.virtual_mem
        self.available_devices[request.id]['battery'] = request.battery
        self.lock.release()

        logging.info(request.id + ': [cpu_usage: ' + str(request.cpu_usage) +
                            ', ncpus: ' + str(request.ncpus) + 
                            ', load: ' + str(request.load15) +
                            ', virtual_mem: ' + str(request.virtual_mem) +
                            ', battery: ' + str(request.battery) +
                            ']')

        return devicetocentral_pb2.Pong(
            ack = True,
        )

 # Loss function

@torch.jit.script
def loss_fn(pred, target):
    return F.nll_loss(input=pred, target=target)

async def fit_model_on_worker(worker, traced_model, batch_size, max_nr_batches, lr, dataset):
    """
    Send the model to the worker and fit the model on the worker's training data.

    Args:
        worker: Remote location, where the model shall be trained.
        traced_model: Model which shall be trained.
        batch_size: Batch size of each training step.
        curr_round: Index of the current training round (for logging purposes).
        max_nr_batches: If > 0, training on worker will stop at min(max_nr_batches, nr_available_batches).
        lr: Learning rate of each training step.

    Returns:
        A tuple containing:
            * worker_id: Union[int, str], id of the worker.
            * improved model: torch.jit.ScriptModule, model after training at the worker.
            * loss: Loss on last training batch, torch.tensor.
    """
    train_config = sy.TrainConfig(
        model=traced_model,
        loss_fn=loss_fn,
        batch_size=batch_size,
        shuffle=True,
        max_nr_batches=max_nr_batches,
        epochs=1,           # only run (1 epoch * batch_size samples) for each round
        optimizer="SGD",
        optimizer_args={"lr": lr},
    )
    start_time = time.time()    # TODO(optional): seperate time spent on async_fit() and network communication
    train_config.send(worker)
    loss = await worker.async_fit(dataset_key=dataset, return_ids=[0])      # TODO: add deadline here
    model = train_config.model_ptr.get().obj
    end_time = time.time()
    duration = end_time - start_time
    return worker.id, model, loss, duration

# async def train_and_eval(worker_instances, args):
#     use_cuda = args.cuda and torch.cuda.is_available()
#     torch.manual_seed(args.seed)
#     device = torch.device("cpu")
#     model = mdlftry.getModel(args.model).to(device)
#     traced_model = torch.jit.trace(model, torch.zeros([1, 3, 32, 32], dtype=torch.float).to(device))
#     learning_rate = args.lr
#     dataset = args.dataset

#     for curr_round in range(1, args.epochs + 1):
#         perf_metrics = recv_perf_metrics(client_list)    #{dev_id: dev_cpu_util}
#         selected_worker_instances = scheduler.rand_scheduler(curr_round, worker_instances, 10)
        
#         results = await asyncio.gather(
#             *[
#                 fit_model_on_worker(worker, traced_model, 128, args.fedepoch, learning_rate, dataset)
#                 for _wi, worker in enumerate(selected_worker_instances) #if batch_size_list[worker.id]>0
#             ]
#         )

#         models = {}
#         loss_values = {}

#         test_models = curr_round % 10 == 1 or curr_round == args.training_rounds

#         max_duration=0
#         # Federate models (note that this will also change the model in models[0]
#         for worker_id, worker_model, worker_loss, duration in results:    # training loss
#             if worker_model is not None:
#                 models[worker_id] = worker_model
#                 loss_values[worker_id] = worker_loss
#                 print(worker_id, duration)
#                 max_duration=max(max_duration, duration)
#                 #if(isstd[worker_id]==True):        # use standard batch size this time -> get dev perf profile
#                 dev_profile[worker_id]=duration

#         print('duration of round ', curr_round, '==', max_duration+schend_time-schstart_time)
#         traced_model = utils.federated_avg(models)

#         test_models=True    # eval after every round
#         sum_acc=0.00
#         if test_models:
#             # evaluate_model_locally(traced_model)
#             for worker in worker_instances:
#                 test_loss, test_acc=evaluate_model_on_worker(        # test accuracy
#                     model_identifier="Federated model",
#                     worker=worker,
#                     dataset_key=dataset + "TEST",
#                     model=traced_model,
#                     nr_bins=10,
#                     batch_size=128,
#                     device="cpu",
#                     print_target_hist=False,
#                 )
#                 evalres[worker.id] = test_acc
#                 evalloss[worker.id] = test_loss
#                 sum_acc+=test_acc
#             sum_acc=sum_acc/len(worker_instances)
#             print("AVG ACCURACY: ",sum_acc)

#         # decay learning rate
#         learning_rate = max(0.98 * learning_rate, args.lr * 0.01)

#     if args.save_model:
#         torch.save(model.state_dict(), "mnist_cnn.pt")

def set_worker_conn(hook, available_devices, verbose):
    worker_instances = {}
    for devid in available_devices:
        kwargs_websocket = {'host' : dev['ip'], 'hook' : hook, 'verbose' : verbose}
        clientWorker = WebsocketClientWorker(id = devid, port = dev['flport'], **kwargs_websocket)
        clientWorker.clear_objects_remote()
        worker_instances.append(clientWorker)
    
    return worker_instances

def schedule_best_worker_instances(available_clients, client_threshold=10, sched_type='PYSched'):
    scheduler = schedftry.getScheduler(sched_type)
    return scheduler.select_worker_instances(available_clients, client_threshold)


def train_and_eval(args, devcentral, client_threshold, verbose):
    use_cuda = args.cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cpu")
    model = mdlftry.getModel(args.model).to(device)
    traced_model = torch.jit.trace(model, torch.zeros([1, 3, 32, 32], dtype=torch.float).to(device))
    max_fed_epoch = args.fedepoch
    learning_rate = args.lr
    dataset = args.dataset
    batch_size = 128

    for curr_round in range(0, args.epochs):
        devcentral.lock.acquire()
        temp_instances = copy.deepcopy(devcentral.available_devices)
        devcentral.lock.release()
        selected_worker_instances = schedule_best_worker_instances(temp_instances, client_threshold)
        
        hook = sy.TorchHook(torch)
        worker_instances = set_worker_conn(hook, selected_worker_instances, verbose)

        results = {}
        for worker in worker_instances:
            results[worker['id']] = fit_model_on_worker(worker, traced_model, batch_size, max_fed_epoch, learning_rate, dataset)

def grpcServe(devcentral):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    devicetocentral_pb2_grpc.add_DeviceToCentralServicer_to_server(devcentral, server)
    server.add_insecure_port('localhost:50051')
    server.start()
    server.wait_for_termination()

def parse_arguments(args = sys.argv[1:]):

    #Parse arguments
    parser = argparse.ArgumentParser(description='Run FL using websocker client workers')

    parser.add_argument(
        '--epochs',
        type = int,
        default = 200,
        help = 'Number of epochs for FL',
    )

    parser.add_argument(
        '--fedepoch',
        type = int,
        default = 10,
        help = 'number of epochs on remote worker before fedavg',
    )

    parser.add_argument(
        '--lr',
        type = float,
        default = 0.1,
        help = 'learning rate',
    )

    parser.add_argument(
        '--cuda',
        action = 'store_true',
        help = 'use cuda if available',
    )

    parser.add_argument(
        '--seed',
        type = int, 
        default = 1,
        help = 'seed for torch random number generator',
    )

    parser.add_argument(
        '--savemodel',
        action = 'store_true',
        help = 'if set, save model'
    )

    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='start websocket client worker in verbose mode: --verbose'
    )

    args = parser.parse_args(args = args)
    return args


if __name__ == '__main__':

    args = parse_arguments()

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S')
    devcentral = DeviceToCentralServicer()
    
    grpcservice = threading.Thread(target=grpcServe, args=(devcentral, ))
    grpcservice.start()

    client_threshold = 10
    train_and_eval(args, devcentral, client_threshold, args.verbose)

    grpcservice.join()
    
