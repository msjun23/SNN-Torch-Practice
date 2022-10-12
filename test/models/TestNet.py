from sklearn import utils
import torch
import torch.nn as nn

import snntorch as snn
from snntorch import surrogate, utils

class TestNet(nn.Module):
    def __init__(self):
        super(TestNet, self).__init__()
        
        # Neuron and simulation params
        spike_grad = surrogate.fast_sigmoid(slope=75)
        beta = 0.5
        
        self.net = nn.Sequential(nn.Conv2d(2, 12, 5), # input ch: 2, output ch: 12, kernel size: 5x5
                    nn.MaxPool2d(2), 
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True), 
                    nn.Conv2d(12, 32, 5), 
                    nn.MaxPool2d(2), 
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True), 
                    nn.Flatten(), 
                    nn.Linear(32*5*5, 10), 
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True))
        
    def forward(self, x):
        spk_rec = []
        utils.reset(self.net)                       # reset hidden states for all LIF neurons in net
        
        for step in range(x.size(0)):               # data.size(0) == # of time steps
            spk_out, mem_out = self.net(x[step])    # for each time step
            spk_rec.append(spk_out)
            
        return torch.stack(spk_rec)
    