
import math

import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.w1 = Parameter(torch.FloatTensor(in_features, out_features))
        self.w2 = Parameter(torch.FloatTensor(in_features, out_features))
        self.w3 = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.w1.size(1))
        self.w1.data.uniform_(-stdv, stdv)
        self.w2.data.uniform_(-stdv, stdv)
        self.w3.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, V, adj):
        o1 = torch.spmm((adj==1).to(torch.float), torch.mm(V, self.w1))
        o2 = torch.spmm((adj==2).to(torch.float), torch.mm(V, self.w2))
        o3 = torch.spmm((adj==3).to(torch.float), torch.mm(V, self.w3))
        output = o1 + o2 + o3
        if self.bias is not None:
            return output + self.bias
        else:
            return output
        
class GraphNet(Module):
    def __init__(self, hidden, fc, pdrop=0.5):
        super(GraphNet, self).__init__()
        
        layers = []
        for i in range(len(hidden)-1):
            a = hidden[i]
            b = hidden[i+1]
            layers.append(GraphConvolution(a,b))
        self._layers = torch.nn.ModuleList(layers)

        full = []
        hfc = [hidden[-1]] + fc
        for i in range(len(hfc)-1):
            a = hfc[i]
            b = hfc[i+1]
            full.append(torch.nn.Linear(a,b))
        self._fc = torch.nn.ModuleList(full)

        self.drop = torch.nn.Dropout(pdrop)
            
    def forward(self, V, adj):
        Z = V
        # graph convolutions
        for l in self._layers:
            Z = F.relu(l(Z, adj))
        # readout
        Z = torch.sum(Z, axis=0)
        # fully connected
        for fc in self._fc[:-1]:
            Z = F.relu(fc(Z))
            Z = self.drop(Z)
        Z = self._fc[-1](Z)
        O = torch.sigmoid(Z)
        return O
    
class GraphNet_old(Module):
    def __init__(self, hidden):
        super(GraphNet_old, self).__init__()
        
        layers = []
        for i in range(len(hidden)-1):
            a = hidden[i]
            b = hidden[i+1]
            layers.append(GraphConvolution(a,b))
        self._layers = torch.nn.ModuleList(layers)
            
    def forward(self, V, adj):
        Z = V
        # graph convolutions
        for l in self._layers:
            Z = F.relu(l(Z, adj))
        # readout
        Z = torch.sum(Z, axis=0)
        O = torch.sigmoid(Z)
        return O
