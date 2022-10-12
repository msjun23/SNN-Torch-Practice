import torch

from snntorch import functional as SF

import matplotlib.pyplot as plt

from models import TestNet
from utils import dataloader

def train(device, net, trainloader, optimizer, loss_fn, epoch):
    loss_hist = []
    acc_hist = []
    epoch_loss = 0
    epoch_acc  = 0

    # training loop
    net.train()
    for batch_idx, (data, targets) in enumerate(iter(trainloader)): # train each batch
        data = data.transpose(0,1)
        data = data.to(device)
        targets = targets.to(device)

        spk_rec = net(data)
        loss = loss_fn(spk_rec, targets)

        # Gradient calculation + weight update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        acc = SF.accuracy_rate(spk_rec, targets)
        
        # Epoch loss & accuracy
        epoch_loss += loss.item()
        epoch_acc  += (acc * 100)

        # Store loss & accuracy history for future plotting
        loss_hist.append(loss.item())
        acc_hist.append(acc)

        if batch_idx % 10 == 0:
            print(f'Epoch {epoch}, Progress {batch_idx/len(trainloader)*100.:.2f}% \nBatch Loss: {loss.item():.2f}')
            print(f'Accuracy: {acc * 100:.2f}%\n')
    
    epoch_loss /= len(trainloader)
    epoch_acc  /= len(trainloader)
    print(f'Epoch {epoch} complete \nAvg. Loss: {epoch_loss:.4f}, Avg. Accuracy: {epoch_acc:.4f} \n\n')
    
    return epoch_loss, epoch_acc

if __name__=='__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)
    net = TestNet.TestNet().to(device)
    trainloader, testloader = dataloader.NMNIST_loader(batch_size=128)
    
    optimizer = torch.optim.Adam(net.parameters(), lr=2e-2, betas=(0.9, 0.999))
    loss_fn = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)

    num_epochs = 10
    loss_hist = []
    acc_hist  = []
    for epoch in range(num_epochs):
        loss, acc = train(device=device, net=net, trainloader=trainloader, optimizer=optimizer, loss_fn=loss_fn, epoch=epoch)
        loss_hist.append(loss)
        acc_hist.append(acc)

    # Plot Loss
    fig = plt.figure(facecolor="w")
    plt.plot(loss_hist)
    plt.title("Train Set Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Avg. Loss")
    
    fig2 = plt.figure(facecolor="w")
    plt.plot(acc_hist)
    plt.title("Train Set Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Avg. Accuracy")
    plt.show()
    