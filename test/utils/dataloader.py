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
    
    trainset = tonic.datasets.NMNIST(save_to='./data', transform=frame_transform, train=True)
    testset  = tonic.datasets.NMNIST(save_to='./data', transform=frame_transform, train=False)
    cached_trainset = CachedDataset(trainset, cache_path='./cache/nmnist/train', transform=aug_transform)
    cached_testset  = CachedDataset(testset, cache_path='./cache/nmnist/test')  # not aug for test set
    
    trainloader = DataLoader(cached_trainset, batch_size=batch_size, collate_fn=tonic.collation.PadTensors(), shuffle=True)
    testloader  = DataLoader(cached_testset, batch_size=batch_size, collate_fn=tonic.collation.PadTensors())
    
    return trainloader, testloader

class DVSCifar10(Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        data, target = torch.load(self.root + '{}.pt'.format(index))

        if self.transform is not None:
            data = self.transform(data)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return data, target.long()

    def __len__(self):
        return len(os.listdir(self.root))

def CIFAR10DVS_loader(batch_size=128):
    sensor_size = tonic.datasets.CIFAR10DVS.sensor_size
    
    frame_transform = transforms.Compose([transforms.Denoise(filter_time=10000), 
                                          transforms.ToFrame(sensor_size=sensor_size, time_window=1000)])
    
    # Augmentation transform
    aug_transform = transforms.Compose([torch.from_numpy, 
                                        torchvision.transforms.RandomRotation([-10,10])])
    
    trainset = tonic.datasets.CIFAR10DVS(save_to='./data', transform=frame_transform)
    # testset  = tonic.datasets.CIFAR10DVS(save_to='./data', transform=frame_transform)
    cached_trainset = CachedDataset(trainset, cache_path='./cache/cifar10dvs/train', transform=aug_transform)
    # cached_testset  = CachedDataset(testset, cache_path='./cache/cifar10dvs/test')  # not aug for test set
    
    trainloader = DataLoader(trainset, batch_size=batch_size, collate_fn=tonic.collation.PadTensors(), shuffle=True)
    # testloader  = DataLoader(cached_testset, batch_size=batch_size, collate_fn=tonic.collation.PadTensors())
    
    return trainloader
    
    # train_dir = 'data/CIFAR10DVS/train'
    # test_dir  = 'data/CIFAR10DVS/test'
    
    # trainloader = DataLoader(DVSCifar10(root=train_dir, transform=frame_transform), 
    #                          batch_size=batch_size, 
    #                          collate_fn=tonic.collation.PadTensors(), 
    #                          shuffle=True)
    # testloader  = DataLoader(DVSCifar10(root=test_dir, transform=frame_transform), 
    #                          batch_size=batch_size, 
    #                          collate_fn=tonic.collation.PadTensors(), 
    #                          shuffle=False)
    
    # return trainloader, testloader
