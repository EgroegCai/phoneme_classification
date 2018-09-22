import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable
from torch.autograd import grad as torch_grad

import numpy as np
import os
import wsj_loader

os.environ['WSJ_PATH'] = 'all'
loader = wsj_loader.WSJ()
trainX, trainY = loader.train
assert(trainX.shape[0] == 24590)

num_environ = 1
freq_size = trainX[0].shape[1]

input_size = freq_size * (1 + num_environ * 2)
output_size = 138
hidden_layers = [2048, 2048, 1024, 1024, 512, 256]

# process training dataset
train_data = []
train_label = []
for utterance in trainX:
    n = len(utterance)
    utterance = np.concatenate(np.zeros((num_environ, freq_size)), utterance)
    utterance = np.concatenate(utterance, np.zeros((num_environ, freq_size)))
    for i in range(num_environ, num_environ+n+1):
        train_data.append(utterance[i-num_environ:i+num_environ+1])

# process training labels
for utterance_label in trainY:
    train_label += list(utterance_label)

train = np.array(zip(train_data, train_label))




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
        return self.model(x)

model = classifier(hidden_layers)
print(model)

lr = 1e-4
betas = (0.5, 0.9)
optimizer = optim.Adam(model.parameters(), lr=lr, betas=betas)

scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

epochs = 10
batch_size = 64

for epoch in range(epochs):
    

    