import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.Dataset as Dataset

from torch.autograd import Variable
from torch.autograd import grad as torch_grad


import numpy as np
import os
import wsj_loader

cuda = torch.cuda.is_available()
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

num_environ = 1
freq_size = trainX[0].shape[1]

input_size = freq_size * (1 + num_environ * 2)
output_size = 138
hidden_layers = [2048, 2048, 1024, 1024, 512, 256]
        

class seqDataSet(Dataset):
    def __init__(self, X, Y, environ):
        self.environ = environ
        self.data = X
        self.label = Y
        self.indices = []
        for i in range(X):
            utterance = X[i]
            n = len(utterance)            
            self.indices += zip(np.full(i), np.arange(n))        

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        seq, pos = self.index[idx]
        utterance = self.data[seq]
        n = len(utterance)
        # retrieve label
        label = self.label[seq][pos]
        y = np.zeros(138)
        y[label] = 1.0

        feed = utterance[max(0, pos-envrion):min(pos+environ+1, n)]
        if pos < environ:
            num_padding = environ - pos
            feed = np.concatenate(np.zeros((num_padding, freq_size)), feed)
        if n - pos - 1 < environ:
            num_padding = environ - (n-pos-1)
            feed = np.concatenate(feed, np.zeros((num_padding, freq_size)))

        feed = torch.tensor(feed).float()
        y = torch.tensor(y).float()
        return feed, y

train_data = seqDataSet(trainX, trainY)
val_data = seqDataSet(devX, devY)

train_dataloader = torch.utils.data.Dataloader(train, batch_size=64,
    shuffle=True, num_workers=8)



# define the MLP model
class classifier(nn.Module):
    def __init__(self, hidden_layers):
        super(classifier, self).__init__()

        nn_layers = []
        nn_layers.append(nn.Linear(input_size, hidden_layers[0]))
        nn_layers.append(nn.BatchNorm1d(hidden_layers[0]))
        nn_layers.append(nn.ReLU())            
        
        for i in range(len(hidden_layers)):
            if i < (len(hidden_layers)-1):
                nn_layers.append(nn.Linear(hidden_layers[i],hidden_layers[i+1]))
                nn_layers.append(nn.BatchNorm1d(hidden_layers[i+1]))
                nn_layers.append(nn.ReLU())
            else:
                nn_layers.append(nn.Linear(hidden_layers[i], output_size))
                nn_layers.append(nn.ReLU())         

        self.model = nn.ModuleList(nn_layers)

    def forward(self, x):
        return self.model(x.view(-1))

model = classifier(hidden_layers)
if cuda:
    model = model.cuda()
print(model)

lr = 1e-4
betas = (0.5, 0.9)
optimizer = optim.Adam(model.parameters(), lr=lr, betas=betas)
criterion = nn.CrossEntropyLoss()

scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

epochs = 10
batch_size = 64

for epoch in range(epochs):
    print('Epoch {}'.format(epoch+1))
    running_loss = 0.0
    tot_match = 0
    for i, data in enumerate(train_dataloader, 0):
        inputs, labels = data

        inputs = Variable(inputs)
        labels = Variable(labels)
        if cuda:
            inputs = inputs.cuda()
            labels = labels.cuda()

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()

        # calculate accuracy
        pred = torch.max(outputs, dim=1)
        truth = torch.max(labels, dim=1)
        num_match = sum((pred == truth) * 1)
        
        tot_match += num_match
        running_loss += loss.item()

        if i % 100:
            # print every 100 iterations
            print('loss: {}'.format(running_loss / 100))
            print('accuracy: {}'.format(tot_match / (100 * batch_size)))
            running_loss = 0.0
            tot_match = 0

torch.save(model.state_dict(), 'model_{}.pt'.format(epoch+1))
    

    