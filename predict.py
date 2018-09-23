import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from torch.autograd import Variable

import argparse
import numpy as np
import os
import wsj_loader
from model import classifier
from dataset import seqDataSet

parser = argparse.ArgumentParser()
parser.add_argument('--save_path', type=str, default='model')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--load', type=bool, default=False)

args = parser.parse_args()
print(args)

cuda = torch.cuda.is_available()
device = torch.device('cuda:0')
print('{} GPUs found.\n'.format(torch.cuda.device_count()))
if cuda:
    print('Using GPU #{}'.format(torch.cuda.current_device()))
else:
    print('GPU not found...process is using CPU.')


os.environ['WSJ_PATH'] = 'all'
loader = wsj_loader.WSJ()
trainX, trainY = loader.train
valX, valY = loader.dev
assert(trainX.shape[0] == 24590)
print(valY.shape[0])

num_environ = 5
freq_size = trainX[0].shape[1]

input_size = freq_size * (1 + num_environ * 2)
output_size = 138
hidden_layers = [2048, 2048, 1024, 1024, 1024, 512]
        
train_data = seqDataSet(trainX, trainY, num_environ)
val_data = seqDataSet(valX, valY, num_environ)

train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True, 
    num_workers=8, drop_last=True)
val_dataloader = DataLoader(val_data, batch_size=64, shuffle=True, 
    num_workers=8, drop_last=True)

path = args.save_path
if not os.path.exists(path):
    os.mkdir(path)

# initialize model            
net = classifier(input_size, output_size, hidden_layers)
if cuda:
    net = net.cuda()
print(net)
if args.load:
    load_path = 'model_beta=0.999_lr=0.001/model_5_125000.pt'
    net.load_state_dict(torch.load(load_path))
    print('model loaded successfully.')


# Testing on using multiple GPUs
if torch.cuda.device_count() > 1:
    net = nn.DataParallel(net)
    net.to(device)
    print('Using {} GPUs'.format(torch.cuda.device_count()))


lr = args.lr
betas = (0.5, 0.999)
optimizer = optim.Adam(net.parameters(), lr=lr, betas=betas)
criterion = nn.CrossEntropyLoss()

scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

epochs = 10
batch_size = 64

# f = open(os.path.join(path, 'accuracy.text'), 'a+')
# f.write('num_environ={} {} beta={} lr='.format(num_environ, hidden_layers, 
#     betas, lr))
# f.flush()
for epoch in range(epochs):
    epoch += 4
    print('Epoch {}'.format(epoch+1))
    running_loss = 0.0
    tot_match = 0
    net.train()
    for i, data in enumerate(train_dataloader, 0):
        inputs, labels = data

        inputs = Variable(inputs)
        labels = Variable(labels)
        if cuda:
            inputs = inputs.cuda()
            labels = labels.cuda()
            if torch.cuda.device_count() > 0:
                inputs = inputs.to(device)
                labels = labels.to(device)

        optimizer.zero_grad()

        outputs = net(inputs.view(batch_size, -1))
        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()

        # calculate accuracy
        pred = torch.argmax(outputs, dim=1)            
        num_match = torch.sum((pred == labels) * 1).item()
        
        tot_match += num_match
        running_loss += loss.item()        
        if (i + 1) % 100 == 0:
            # print every 100 iterations            
            print('iteration {}'.format(i))
            print('training loss: {}'.format(running_loss / 100))
            print('training accuracy: {}'.format(tot_match / (100 * batch_size)))            
            running_loss = 0.0
            tot_match = 0

        # validation        
        if (i+1) % 10000 == 0:
            net.eval()
            with torch.no_grad():
                val_running_loss = 0.0
                total = 0
                correct = 0
                for _,vdata in enumerate(val_dataloader,0):
                    vinputs, vlabels = vdata
                    vinputs = Variable(vinputs)
                    vlabels = Variable(vlabels)

                    if cuda:
                        vinputs = vinputs.cuda()
                        vlabels = vlabels.cuda()                    
                    voutputs = net(vinputs.view(batch_size, -1))
                    vloss = criterion(voutputs, vlabels)
                    val_running_loss += vloss.item()

                    vpred = torch.argmax(voutputs, dim=1)                                                    
                    correct += torch.sum((vpred == vlabels) * 1).item()
                    total += vlabels.size(0)
            print()                     
            print('validation loss: {}'.format(val_running_loss / (total / batch_size)))
            print('validation accuracy: {}'.format(correct / total))
            print()
            # f.write('{}\n'.format(correct / total))
            # f.flush()

        if i % 5000 == 0:
            torch.save(net.state_dict(), os.path.join(path, 'model_{}_{}.pt'.format(epoch+1, i)))
    

    