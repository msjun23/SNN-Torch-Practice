import tonic
import tonic.transforms as transforms
from tonic import CachedDataset

import numpy as np
import os
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset

def NMNIST_loader(batch_size=128):
    sensor_size = tonic.datasets.NMNIST.sensor_size
    
    # time_window=1000 integrates events into 1000us bins
    # Denoise removes isolated, one-off events.
    # If no event occurs within a neighbourhood of 1 pixel across filter_time microseconds(us), 
    # the event is filtered. Smaller filter_time will filter more events.
    frame_transform = transforms.Compose([transforms.Denoise(filter_time=10000), 
                                          transforms.ToFrame(sensor_size=sensor_size, time_window=1000)])
    # Augmentation transform
    aug_transform = transforms.Compose([torch.from_numpy, 
                                        torchvision.transforms.RandomRotation([-10,10])])
    
    # [(x,y,timestamp,polarity), ...]
    trainset = tonic.datasets.NMNIST(save_to='./data', transform=frame_transform, train=True)
    testset  = tonic.datasets.NMNIST(save_to='./data', transform=frame_transform, train=False)
    cached_trainset = CachedDataset(trainset, cache_path='./cache/nmnist/train', transform=aug_transform)
    cached_testset  = CachedDataset(testset, cache_path='./cache/nmnist/test')  # not aug for test set
    
    trainloader = DataLoader(cached_trainset, batch_size=batch_size, collate_fn=tonic.collation.PadTensors(), shuffle=True)
    testloader  = DataLoader(cached_testset, batch_size=batch_size, collate_fn=tonic.collation.PadTensors())
    
    return trainloader, testloader

def POKERDVS_loader(batch_size=128):
    sensor_size = tonic.datasets.POKERDVS.sensor_size
    
    frame_transform = transforms.Compose([transforms.Denoise(filter_time=10000), 
                                          transforms.ToFrame(sensor_size=sensor_size, time_window=1000)])
    # Augmentation transform
    aug_transform = transforms.Compose([torch.from_numpy, 
                                        torchvision.transforms.RandomRotation([-10,10])])
    
    # [(x,y,timestamp,polarity), ...]
    trainset = tonic.datasets.POKERDVS(save_to='./data', transform=frame_transform, train=True)
    testset  = tonic.datasets.POKERDVS(save_to='./data', transform=frame_transform, train=False)
    cached_trainset = CachedDataset(trainset, cache_path='./cache/pokerdvs/train', transform=aug_transform)
    cached_testset  = CachedDataset(testset, cache_path='./cache/pokerdvs/test')  # not aug for test set
    
    trainloader = DataLoader(cached_trainset, batch_size=batch_size, collate_fn=tonic.collation.PadTensors(), shuffle=True)
    testloader  = DataLoader(cached_testset, batch_size=batch_size, collate_fn=tonic.collation.PadTensors())
    
    return trainloader, testloader

def DVSGesture_loader(batch_size=128):
    sensor_size = tonic.datasets.DVSGesture.sensor_size
    
    frame_transform = transforms.Compose([transforms.Denoise(filter_time=10000), 
                                          transforms.ToFrame(sensor_size=sensor_size, time_window=1000)])
    # Augmentation transform
    aug_transform = transforms.Compose([torch.from_numpy, 
                                        torchvision.transforms.RandomRotation([-10,10])])
    
    # [(x,y,timestamp,polarity), ...]
    trainset = tonic.datasets.DVSGesture(save_to='./data', transform=frame_transform, train=True)
    testset  = tonic.datasets.DVSGesture(save_to='./data', transform=frame_transform, train=False)
    cached_trainset = CachedDataset(trainset, cache_path='./cache/dvsgesture/train', transform=aug_transform)
    cached_testset  = CachedDataset(testset, cache_path='./cache/dvsgesture/test')  # not aug for test set
    
    trainloader = DataLoader(cached_trainset, batch_size=batch_size, collate_fn=tonic.collation.PadTensors(), shuffle=True)
    testloader  = DataLoader(cached_testset, batch_size=batch_size, collate_fn=tonic.collation.PadTensors())
    
    return trainloader, testloader

def DVSLip_loader(batch_size=128):
    sensor_size = tonic.datasets.DVSLip.sensor_size
    
    frame_transform = transforms.Compose([transforms.Denoise(filter_time=10000), 
                                          transforms.ToFrame(sensor_size=sensor_size, time_window=1000)])
    # Augmentation transform
    aug_transform = transforms.Compose([torch.from_numpy, 
                                        torchvision.transforms.RandomRotation([-10,10])])
    
    # [(x,y,timestamp,polarity), ...]
    trainset = tonic.datasets.DVSLip(save_to='./data', transform=frame_transform, train=True)
    testset  = tonic.datasets.DVSLip(save_to='./data', transform=frame_transform, train=False)
    cached_trainset = CachedDataset(trainset, cache_path='./cache/dvslip/train', transform=aug_transform)
    cached_testset  = CachedDataset(testset, cache_path='./cache/dvslip/test')  # not aug for test set
    
    trainloader = DataLoader(cached_trainset, batch_size=batch_size, collate_fn=tonic.collation.PadTensors(), shuffle=True)
    testloader  = DataLoader(cached_testset, batch_size=batch_size, collate_fn=tonic.collation.PadTensors())
    
    return trainloader, testloader

def NCALTECH101_loader(batch_size=128):
    sensor_size = tonic.datasets.NCALTECH101.sensor_size
    
    # To frame transform
    frame_transform = transforms.Compose([transforms.Denoise(filter_time=10000), 
                                          transforms.ToFrame(sensor_size=sensor_size, time_window=1000)])
    # Augmentation transform
    aug_transform = transforms.Compose([torch.from_numpy, 
                                        torchvision.transforms.RandomRotation([-10,10])])
    
    # [(timestamp,x,y,polarity), ...]
    dataset = tonic.datasets.NCALTECH101(save_to='./data', transform=frame_transform)
    cached_dataset = CachedDataset(dataset, cache_path='./cache/ncaltech101', transform=aug_transform)
    
    # Split into train & test
    train_size = int(len(cached_dataset) * 0.7)
    test_size  = int(len(cached_dataset) * 0.3)
    trainset, testset = torch.utils.data.random_split(cached_dataset, [train_size, test_size])
    
    trainloader = DataLoader(trainset, batch_size=batch_size, collate_fn=tonic.collation.PadTensors(), shuffle=True)
    testloader  = DataLoader(testset, batch_size=batch_size, collate_fn=tonic.collation.PadTensors())
    
    return trainloader, testloader

def ASLDVS_loader(batch_size=128):
    sensor_size = tonic.datasets.ASLDVS.sensor_size
    
    # To frame transform
    frame_transform = transforms.Compose([transforms.Denoise(filter_time=10000), 
                                          transforms.ToFrame(sensor_size=sensor_size, time_window=1000)])
    # Augmentation transform
    aug_transform = transforms.Compose([torch.from_numpy, 
                                        torchvision.transforms.RandomRotation([-10,10])])
    
    # [(timestamp,x,y,polarity), ...]
    dataset = tonic.datasets.ASLDVS(save_to='./data', transform=frame_transform)
    cached_dataset = CachedDataset(dataset, cache_path='./cache/asldvs', transform=aug_transform)
    
    # Split into train & test
    train_size = int(len(cached_dataset) * 0.7)
    test_size  = int(len(cached_dataset) * 0.3)
    trainset, testset = torch.utils.data.random_split(cached_dataset, [train_size, test_size])
    
    trainloader = DataLoader(trainset, batch_size=batch_size, collate_fn=tonic.collation.PadTensors(), shuffle=True)
    testloader  = DataLoader(testset, batch_size=batch_size, collate_fn=tonic.collation.PadTensors())
    
    return trainloader, testloader

def CIFAR10DVS_loader(batch_size=128):
    sensor_size = tonic.datasets.CIFAR10DVS.sensor_size
    
    # To frame transform
    frame_transform = transforms.Compose([transforms.Denoise(filter_time=10000), 
                                          transforms.ToFrame(sensor_size=sensor_size, time_window=1000)])
    # Augmentation transform
    aug_transform = transforms.Compose([torch.from_numpy, 
                                        torchvision.transforms.RandomRotation([-10,10])])
    
    # [(timestamp,x,y,polarity), ...]
    dataset = tonic.datasets.CIFAR10DVS(save_to='./data', transform=frame_transform)
    cached_dataset = CachedDataset(dataset, cache_path='./cache/cifar10dvs', transform=aug_transform)
    
    # Split into train & test
    train_size = int(len(cached_dataset) * 0.7)
    test_size  = int(len(cached_dataset) * 0.3)
    trainset, testset = torch.utils.data.random_split(cached_dataset, [train_size, test_size])
    
    trainloader = DataLoader(trainset, batch_size=batch_size, collate_fn=tonic.collation.PadTensors(), shuffle=True)
    testloader  = DataLoader(testset, batch_size=batch_size, collate_fn=tonic.collation.PadTensors())
    
    return trainloader, testloader
