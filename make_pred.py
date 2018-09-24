import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torch.autograd import Variable

import argparse
import numpy as np
import os
import wsj_loader
from model import classifier
from dataset import seqDataSet

cuda = torch.cuda.is_available()
print('{} GPUs found.\n'.format(torch.cuda.device_count()))
if cuda:
    print('Using GPU #{}'.format(torch.cuda.current_device()))
else:
    print('GPU not found...process is using CPU.')


os.environ['WSJ_PATH'] = 'all'
loader = wsj_loader.WSJ()
testX, testY = loader.test


num_environ = 5
freq_size = 40

input_size = freq_size * (1 + num_environ * 2)
output_size = 138
hidden_layers = [2048, 2048, 1024, 1024, 1024, 512]

model_path = 'model_beta=0.999_no_dropout/model_20.pt'

net = classifier(input_size, output_size, hidden_layers)

if cuda:
    net = net.cuda()
    
if torch.cuda.device_count() > 0:
    device = torch.device('cuda:0')
    net = nn.DataParallel(net)
    net.to(device)
net.load_state_dict(torch.load(model_path))


test_data = seqDataSet(testX, testY, num_environ)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False, 
    num_workers=8, drop_last=False)

f = open('prediction5.csv', 'w+')
f.write('id,label\n')
with torch.no_grad():
    net.eval()
    for i, data in enumerate(test_loader):
        inputs,_ = data
        inputs = Variable(inputs)
        if cuda:
            inputs = inputs.cuda()

        output = net(inputs.view(inputs.size(0),-1))
        label = torch.argmax(output, 1)
        
        for idx in range(inputs.size(0)):
            print('id = {}, prediction = {}'.format(i*64+idx, label[idx]))
            f.write('{},{}\n'.format(i*64+idx, label[idx]))
            f.flush()


