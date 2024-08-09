import torch
import torch.nn as nn

from torch_geometric.nn.conv import HeteroConv, SAGEConv
from hedge.utils import reverse_edge


class NHP(torch.nn.Module):
    """
    Neural Hyperlink Predictor
    """

    def __init__(self, in_channels,out_channels):
        super().__init__()

        self.layers = torch.nn.ModuleList()
        self.layers.append(HeteroConv(
            {('node','in','hyperedge'): SAGEConv(in_channels,out_channels,add_self_loops = False)
            },aggr = 'sum'))
            
    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, data):
        
        device = data['node'].x.device
        x_dict = {'node':data['node'].x,'hyperedge': torch.zeros((data['hyperedge'].num_nodes, data['node'].x.shape[1])).to(device)}
        edge_index_dict = {
            ('node','in','hyperedge'):data.edge_index_dict['node','in','hyperedge'],
            # ('hyperedge','rev_in','node'):reverse_edge(data.edge_index_dict['node','in','hyperedge'])
        }
        
        for layer in self.layers:
            x_dict = layer(x_dict,edge_index_dict)
        print(len(x_dict))
        
        
        return x_dict['hyperedge']
    
    def _mean(self, K, H):
        L = K*K
        L = L/torch.sum(L, dim=0)
        return torch.mm(L.t(), H)


    def _maxmin(self, K, H):
        L = K.t()

        B = H.repeat(L.size()[0], 1, 1)
        d = B.size()[-1]
        L = L.repeat_interleave(d).view(L.size()[0], L.size()[1], d)

        LB = (L == 1).float()*B
        M = torch.max(LB, dim = 1)[0]

        if self.Type == "d": LB = (L == -1).float()*B
        m = torch.min((LB==0).float()*10000 + LB, dim=1)[0]

        return (M - m)*((M-m>0).float())


    def _get(self, data, k): 
        if self.test == True: return data[k][0].to('cpu')
        return data[k][0].to(self.device)