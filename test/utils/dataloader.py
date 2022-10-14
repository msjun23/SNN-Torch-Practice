import tonic
import tonic.transforms as transforms
from tonic import CachedDataset

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader

def NMNIST_loader(batch_size=128):
    sensor_size = tonic.datasets.NMNIST.sensor_size
    
    # time_window=1000 integrates events into 1000s bins
    # Denoise removes isolated, one-off events.
    # If no event occurs within a neighbourhood of 1 pixel across filter_time microseconds, 
    # the event is filtered. Smaller filter_time will filter more events.
    frame_transform = transforms.Compose([transforms.Denoise(filter_time=10000), 
                                         transforms.ToFrame(sensor_size=sensor_size, time_window=1000)])
    # Augmentation transform
    aug_transform = transforms.Compose([torch.from_numpy, 
                                        torchvision.transforms.RandomRotation([-10,10])])
    
    trainset = tonic.datasets.NMNIST(save_to='./data', transform=frame_transform, train=True)
    testset  = tonic.datasets.NMNIST(save_to='./data', transform=frame_transform, train=False)
    cached_trainset = CachedDataset(trainset, cache_path='./cache/nmnist/train', transform=aug_transform)
    cached_testset  = CachedDataset(testset, cache_path='./cache/nmnist/test')  # not aug for test set
    
    trainloader = DataLoader(cached_trainset, batch_size=batch_size, collate_fn=tonic.collation.PadTensors(), shuffle=True)
    testloader  = DataLoader(cached_testset, batch_size=batch_size, collate_fn=tonic.collation.PadTensors())
    
    return trainloader, testloader

def CIFAR10DVS_loader(batch_size=128):
    sensor_size = tonic.datasets.CIFAR10DVS.sensor_size
    
    frame_transform = transforms.Compose([transforms.Denoise(filter_time=10000), 
                                          transforms.ToFrame(sensor_size=sensor_size, time_window=1000)])
    
    # Augmentation transform
    aug_transform = transforms.Compose([torch.from_numpy, 
                                        torchvision.transforms.RandomRotation([-10,10])])
    
    # trainset = tonic.datasets.CIFAR10DVS(save_to='./data', transform=frame_transform)
    trainset = tonic.datasets.CIFAR10DVS(save_to='./data')
    # testset  = tonic.datasets.CIFAR10DVS(save_to='./data', transform=frame_transform)
    cached_trainset = CachedDataset(trainset, cache_path='./cache/cifar10dvs/train', transform=aug_transform)
    # cached_testset  = CachedDataset(testset, cache_path='./cache/cifar10dvs/test')  # not aug for test set
    
    trainloader = DataLoader(cached_trainset, batch_size=batch_size, collate_fn=tonic.collation.PadTensors(), shuffle=True)
    # testloader  = DataLoader(cached_testset, batch_size=batch_size, collate_fn=tonic.collation.PadTensors(), shuffle=True)
    
    return trainloader
