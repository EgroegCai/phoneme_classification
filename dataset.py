import torch
from torch.utils.data import Dataset

import numpy as np

# class seqDataSet(Dataset):
#     def __init__(self, X, Y, environ):
#         freq_size = 40
#         self.X = []
#         for utterance in X:
#             n = len(utterance)
#             for i in range(n):
#                 feed = utterance[max(0, i-environ):min(i+environ+1, n)]
#                 if i < environ:
#                     num_padding = environ - i
#                     padding = np.zeros((num_padding, freq_size))                        
#                     feed = np.concatenate((padding, feed))
#                 if n - i - 1 < environ:
#                     num_padding = environ - (n-i-1)
#                     padding = np.zeros((num_padding, freq_size))            
#                     feed = np.concatenate((feed, padding))
#                 self.X.append(feed)
#         self.X = np.array(self.X)

#         if Y != None:
#             self.Y = []
#             for utterance_label in Y:
#                 self.Y += list(utterance_label)
#             self.Y = np.array(self.Y)

#     def __len__(self):
#         return len(self.X)

#     def __getitem__(self, idx):
#         data = torch.tensor(self.X[idx]).float()
#         if Y != None:
#             label = self.Y[idx]
#         else:
#             label = 0

#         return data, label

class seqDataSet(Dataset):
    def __init__(self, X, Y, environ):
        self.freq_size = 40
        self.environ = environ
        self.data = X
        self.label = Y
        self.indices = []
        for i in range(X.shape[0]):
            utterance = X[i]
            n = len(utterance)            
            self.indices += zip(np.full(n, i), np.arange(n))  
        print(len(self.indices))     

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):        
        environ = self.environ
        seq, pos = self.indices[idx]
        utterance = self.data[seq]
        n = len(utterance)

        feed = utterance[max(0, pos-environ):min(pos+environ+1, n)]
        if pos < environ:
            num_padding = environ - pos
            padding = np.zeros((num_padding, self.freq_size))                        
            feed = np.concatenate((padding, feed))
        if n - pos - 1 < environ:
            num_padding = environ - (n-pos-1)
            padding = np.zeros((num_padding, self.freq_size))            
            feed = np.concatenate((feed, padding))

        feed = torch.tensor(feed).float()
        # retrieve label
        if self.label != None:
            label = self.label[seq][pos]                        
        else:
            label = 0
        return feed, label