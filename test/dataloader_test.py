import tonic
import tonic.transforms as transforms
from tonic import CachedDataset

import torch
import torchvision
from torch.utils.data import DataLoader, Dataset

import numpy as np

if __name__=='__main__':
    nmnist = tonic.datasets.NMNIST(save_to='./data', train=True)
    events, target = nmnist[0]
    # [(x,y,timestamp,polarity), ...]
    print('\n### N-MNIST ###')
    print('* dataset: ', nmnist, '/ dataset len: ', len(nmnist), type(nmnist))
    print('* events : ', events, '/ events len: ', len(events), '/ shape & type', np.shape(events), type(events))
    print('* target : ', target, type(target))
    tonic.utils.plot_event_grid(events=events)
    
    print('\n---\n')
    
    pokerdvs = tonic.datasets.POKERDVS(save_to='./data', train=True)
    events, target = pokerdvs[0]
    # [(timestep,x,y,polarity), ...]
    print('\n### POKER-DVS ###')
    print('* dataset: ', pokerdvs, '/ dataset len: ', len(pokerdvs), type(pokerdvs))
    print('* events : ', events, '/ events len: ', len(events), '/ shape & type', np.shape(events), type(events))
    print('* target : ', target, type(target))
    tonic.utils.plot_event_grid(events=events)
    
    print('\n---\n')
    
    dvsgesture = tonic.datasets.DVSGesture(save_to='./data', train=True)
    events, target = dvsgesture[0]
    # [(x,y,polarity,timestamp), ...]
    print('\n### IBM DVS Gesture ###')
    print('* dataset: ', dvsgesture, '/ dataset len: ', len(dvsgesture), type(dvsgesture))
    print('* events : ', events, '/ events len: ', len(events), '/ shape & type', np.shape(events), type(events))
    print('* target : ', target, type(target))
    tonic.utils.plot_event_grid(events=events)
    
    print('\n---\n')
    
    dvslip = tonic.datasets.DVSLip(save_to='./data', train=True)
    events, target = dvslip[0]
    # [(x,y,polarity,timestamp), ...]
    print('\n### DVS-Lip ###')
    print('* dataset: ', dvslip, '/ dataset len: ', len(dvslip), type(dvslip))
    print('* events : ', events, '/ events len: ', len(events), '/ shape & type', np.shape(events), type(events))
    print('* target : ', target, type(target))
    tonic.utils.plot_event_grid(events=events)
    
    print('\n---\n')
    
    ncaltech = tonic.datasets.NCALTECH101(save_to='./data')
    events, target = ncaltech[7891]
    # [(x,y,timestamp,polarity), ...]
    print('\n### N-CALTECH101 ###')
    print('* dataset: ', ncaltech, '/ dataset len: ', len(ncaltech), type(ncaltech))
    print('* events : ', events, '/ events len: ', len(events), '/ shape & type', np.shape(events), type(events))
    print('* target : ', target, type(target))
    tonic.utils.plot_event_grid(events=events)
    
    print('\n---\n')
    
    asldvs = tonic.datasets.ASLDVS(save_to='./data')
    events, target = asldvs[100000]
    events = np.squeeze(events)
    # [(timestamp,x,y,polarity), ...]
    print('\n### ASL-DVS ###')
    print('* dataset: ', asldvs, '/ dataset len: ', len(asldvs), type(asldvs))
    print('* events : ', events, '/ events len: ', len(events), '/ shape & type', np.shape(events), type(events))
    print('* target : ', target, type(target))
    tonic.utils.plot_event_grid(events=events)
    
    print('\n---\n')
    
    cifar10dvs = tonic.datasets.CIFAR10DVS(save_to='./data')
    events, target = cifar10dvs[0]
    # [(timestamp,x,y,polarity), ...]
    print('\n### CIFAR10-DVS ###')
    print('* dataset: ', cifar10dvs, '/ dataset len: ', len(cifar10dvs), type(cifar10dvs))
    print('* events : ', events, '/ events len: ', len(events), '/ shape & type', np.shape(events), type(events))
    print('* target : ', target, type(target))
    tonic.utils.plot_event_grid(events=events)
    
    print('\n---\n')
    