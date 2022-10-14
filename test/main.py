import torch
import numpy as np
from tqdm import tqdm

from snntorch import functional as SF

import matplotlib.pyplot as plt

from models.TestNet import *
from utils import dataloader

def Train(device, net, trainloader, optimizer, loss_fn, epoch):
    epoch_loss = 0
    epoch_acc  = 0

    # training loop
    net.train()
    print(f'Epoch {epoch}')
    for data, targets in tqdm(iter(trainloader)): # train each batch
        data = data.transpose(0,1)
        data = data.to(device)
        targets = targets.to(device)

        # Forward pass
        spk_rec = net(data)
        loss = loss_fn(spk_rec, targets)

        # Gradient calculation + weight update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        acc = SF.accuracy_rate(spk_rec, targets)
        
        # Epoch loss & accuracy
        epoch_loss += loss.item()       # sum up batch loss
        epoch_acc  += (acc * 100)       # sum up batch accuracy

    epoch_loss /= len(trainloader)
    epoch_acc  /= len(trainloader)
    print(f'Train set, Avg. Loss: {epoch_loss:.4f}, Avg. Accuracy: {epoch_acc:.4f}')
    
    return epoch_loss, epoch_acc

def Test(device, net, testloader, loss_fn):
    test_loss = 0
    test_acc  = 0
    
    net.eval()
    with torch.no_grad():
        for data, targets in tqdm(iter(testloader)):       # test each batch
            data = data.transpose(0,1)
            data = data.to(device)
            targets = targets.to(device)
            
            # Test forward pass
            spk_rec = net(data)
            
            # Calculate loss & accuracy
            test_loss += loss_fn(spk_rec, targets).item()
            _, idx   = spk_rec.sum(dim=0).max(1)
            test_acc += (np.mean((targets==idx).detach().cpu().numpy()) * 100)
            
    test_loss /= len(testloader)
    test_acc  /= len(testloader)
    print(f'Test set, Avg. Loss: {test_loss:.4f}, Avg. Accuracy: {test_acc:.4f} \n\n')
    
    return test_loss, test_acc
    

if __name__=='__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('device: ', device)
    net = TestNet().to(device)
    trainloader, testloader = dataloader.NMNIST_loader(batch_size=128)
    # trainloader = dataloader.CIFAR10DVS_loader(batch_size=128)
    
    optimizer = torch.optim.Adam(net.parameters(), lr=2e-2, betas=(0.9, 0.999))
    loss_fn = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)

    num_epochs = 10
    loss_hist = []
    acc_hist  = []
    test_loss_hist = []
    test_acc_hist  = []
    for epoch in range(num_epochs):
        # Train
        loss, acc = Train(device=device, net=net, trainloader=trainloader, optimizer=optimizer, loss_fn=loss_fn, epoch=epoch)
        loss_hist.append(loss)
        acc_hist.append(acc)
        
        # Test
        test_loss, test_acc = Test(device=device, net=net, testloader=testloader, loss_fn=loss_fn)
        test_loss_hist.append(test_loss)
        test_acc_hist.append(test_acc)

    # Plot result
    fig = plt.figure(facecolor="w")
    plt.plot(loss_hist)
    plt.plot(test_loss_hist)
    plt.title("Train & Test Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Avg. Loss")
    plt.legend(['Train loss', 'Test loss'])
    
    fig2 = plt.figure(facecolor="w")
    plt.plot(acc_hist)
    plt.plot(test_acc_hist)
    plt.title("Train & Test Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Avg. Accuracy")
    plt.legend(['Train acc', 'Test acc'])
    plt.show()
    