import torch
# from torch_geometric.nn.conv import GraphConv
from hedge.utils import reverse_edge
from hedge.layer import BertConv



# from typing import Tuple, Union
# from torch import Tensor

from torch_geometric.nn.conv import HeteroConv, SAGEConv, GATConv, GINConv ,GINEConv, ResGatedGraphConv,GATv2Conv,TransformerConv,SimpleConv,GCNConv,GraphConv,MFConv,GMMConv,SplineConv
from torch_geometric.nn.conv import NNConv,CGConv,EdgeConv,FeaStConv,LEConv,GENConv,GeneralConv
# from torch_geometric.nn.dense.linear import Linear
# from torch_geometric.typing import (
#     Adj,
#     OptPairTensor,
#     OptTensor,
#     Size,
#     SparseTensor,
# )
from torch_geometric.utils import spmm

        




def get_conv_layer(conv,hidden_channels,out_channels,add_self_loops = False, heads = 1, dropout = 0.0,nlp_model = None):       
    
    conv = conv.lower()
    conv_dict  = {'sage':SAGEConv,'gat':GATConv,'gin':GINConv,
                'gine':GINEConv,'resgatedgraph':ResGatedGraphConv,
              'gatv2':GATv2Conv,'transformer':TransformerConv,
              'simple':SimpleConv,'gcn':GCNConv,'graph':GraphConv,
              'mf':MFConv,'gmm':GMMConv,'spline':SplineConv,
              'nn':NNConv,'cg':CGConv,'edge':EdgeConv,
              'feast':FeaStConv,'le':LEConv,'gen':GENConv,'general':GeneralConv,
              'bert':BertConv}
    
    if conv not in conv_dict:
        raise NotImplementedError('conv {} is not implemented'.format(conv))
        
    layer =  conv_dict[conv]
    if conv in set(['transformer','gatv2','gat']):
        return layer(hidden_channels,out_channels,add_self_loops=add_self_loops, heads = heads, dropout = dropout, concat = False)
    elif conv in set(['simple']):
        return layer(hidden_channels,out_channels,add_self_loops=add_self_loops, dropout = dropout, combine_root = 'sum')
    elif conv in set(['bert']):
        return layer(nlp_model)
    else:
        # return layer(hidden_channels,out_channels,add_self_loops=add_self_loops, dropout = dropout)
        # return layer(hidden_channels,out_channels, dropout = dropout)
        return layer(hidden_channels,out_channels)


        
        

class BaseModel(torch.nn.Module):

    def __init__(self, in_channels,out_channels,hidden_channels =64, conv = 'SAGEConv', aggr = 'mean', bias = True, num_layers = 1, dropout = 1.0, heads = 1, nlp_model = None):
        super().__init__()

        self.dropout = dropout
        self.layers = torch.nn.ModuleList()
        self.aggr = aggr
        self.bias = bias
        self.num_layers = num_layers
        
        self.dropout = torch.nn.Dropout(dropout)
        self.bn_layers = torch.nn.ModuleList()
        
        # self.conv = conv

        if self.num_layers ==1:
            n_v = get_conv_layer(conv,in_channels,out_channels,heads = heads, dropout = dropout, nlp_model = nlp_model)
            v_n = get_conv_layer(conv,in_channels,out_channels,heads = heads, dropout = dropout, nlp_model = nlp_model)
            self.layers.append(HeteroConv({('node','in','hyperedge'): n_v,('hyperedge','rev_in','node'): v_n},aggr = 'sum'))
            self.bn_layers.append(torch.nn.BatchNorm1d(out_channels))
            
        else:
            n_v = get_conv_layer(conv,in_channels,hidden_channels,heads = heads, dropout = dropout, nlp_model = nlp_model)
            v_n = get_conv_layer(conv,in_channels,hidden_channels,heads = heads, dropout = dropout, nlp_model = nlp_model)
            self.layers.append(HeteroConv({('node','in','hyperedge'): n_v,('hyperedge','rev_in','node'): v_n},aggr = 'sum'))
            self.bn_layers.append(torch.nn.BatchNorm1d(hidden_channels))
            
            for _ in range(self.num_layers - 2):
                
                n_v = get_conv_layer(conv,hidden_channels,hidden_channels,heads = heads, dropout = dropout, nlp_model = nlp_model)
                v_n = get_conv_layer(conv,hidden_channels,hidden_channels,heads = heads, dropout = dropout, nlp_model = nlp_model)
                self.layers.append(HeteroConv({('node','in','hyperedge'): n_v,('hyperedge','rev_in','node'): v_n},aggr = 'sum'))  
                self.bn_layers.append(torch.nn.BatchNorm1d(hidden_channels))
                

            n_v = get_conv_layer(conv,hidden_channels,out_channels,heads = heads, dropout = dropout, nlp_model = nlp_model)
            v_n = get_conv_layer(conv,hidden_channels,out_channels,heads = heads, dropout = dropout, nlp_model = nlp_model)
            self.layers.append(HeteroConv({('node','in','hyperedge'): n_v,('hyperedge','rev_in','node'): v_n},aggr = 'sum'))
            self.bn_layers.append(torch.nn.BatchNorm1d(out_channels))
            
            
                
    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, data):
        
        device = data['node'].x.device
        x_dict = {'node':data['node'].x,'hyperedge': torch.zeros((data['hyperedge'].num_nodes, data['node'].x.shape[1])).to(device)}
        edge_index_dict = {
            ('node','in','hyperedge'):data.edge_index_dict['node','in','hyperedge'],
            ('hyperedge','rev_in','node'):reverse_edge(data.edge_index_dict['node','in','hyperedge'])
        }
        
        for i, layer in  enumerate(self.layers):
            x_dict = layer(x_dict,edge_index_dict)
            
            # for key in x_dict.keys():
            #     x_dict[key] = self.bn_layers[i](x_dict[key])

            if i != self.num_layers -1:
                # x_dict = {k:torch.relu(v) for k,v in x_dict.items()}
                x_dict = {k:self.dropout(v) for k,v in x_dict.items()}
             
        return x_dict
    
    
    

    
    """
    x0       
    y0
    
    x1 = x0 * w11 + G * y0 * w12  = x0 * w11 
    y1 = y0 * w11 + G * x0 * w12  = G * x0 * w12
    
    x2 = x1 * w21 + G * y1 * w22  = x0 * w11 * w21       + G * G * x0 * w12 * w22
    y2 = y1 * w21 + G * x1 * w22  = G  * x0  * w12 * w22 + G *     x0 * w11 * w22 = G * x0 * (w11 * w22 + w12 * w21)
    
    
    """
     
