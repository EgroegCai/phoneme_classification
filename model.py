import torch
import torch.nn as nn

# define the MLP model
class classifier(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers):
        super(classifier, self).__init__()

        nn_layers = []
        nn_layers.append(nn.Linear(input_size, hidden_layers[0]))
        nn_layers.append(nn.BatchNorm1d(hidden_layers[0]))
        nn_layers.append(nn.ReLU())            
        
        for i in range(len(hidden_layers)):
            if i < (len(hidden_layers)-1):
                # self.f = nn.Linear(hidden_layers[i],hidden_layers[i+1])
                # nn.init.xavier_uniform_(self.f.weight.data, 1.)
                nn_layers.append(nn.Linear(hidden_layers[i],hidden_layers[i+1]))
                nn_layers.append(nn.BatchNorm1d(hidden_layers[i+1]))
                nn_layers.append(nn.ReLU())
            else:
                nn_layers.append(nn.Linear(hidden_layers[i], output_size))
                # nn_layers.append(nn.ReLU())         

        self.model = nn.ModuleList(nn_layers)        

    def forward(self, x):        
        for fc in self.model:            
            x = fc(x)
        return x

