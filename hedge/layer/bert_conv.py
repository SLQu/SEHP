

from typing import Tuple, Union, List
from torch import Tensor

import torch
from torch_geometric.nn.conv import MessagePassing,HeteroConv, SAGEConv
from torch_geometric.nn.dense.linear import Linear

from torch_geometric.typing import (
    Adj,
    OptPairTensor,
    OptTensor,
    Size,
    SparseTensor
)
from torch_geometric.utils import spmm


from typing import Optional

# BertAggregation

from torch_geometric.nn.aggr import Aggregation
from hedge.aggregator import BertAggregation
    

class BertConv(MessagePassing):
    def __init__(
        self,
        nlp_model,
        **kwargs,
    ):
        
        aggr = BertAggregation(nlp_model)
        super().__init__(aggr=aggr, **kwargs)

        self.nlp_model = nlp_model
        
        # self.reset_parameters()
        
    def forward(self,x: Union[Tensor, OptPairTensor], edge_index: Adj, size: Size = None):
        
        if isinstance(x, Tensor):
            x = (x, x)
        
        return self.propagate(edge_index, x=x, size=size)
        
    def message(self, x_j):
        return x_j


    def reset_parameters(self):
        super().reset_parameters()
        
        
        """
在图表征学习中，信息通过一个节点流到另外一个节点上，这就是一个encoder的过程，得到节点的最终表征。
然而超图中，信息的传播是从节点流到超边，通过encoder得到超边的表征；然后再从超边流到节点，通过encoder得到节点的表征。

我现在想做的任务是这样的：
1. 现在有一个超图，每个节点都是一个token，节点之间没有顺序关系，多个节点成一个超边。我想使用预训练好的Bert模型作为encoder。最初的输入是token的原始embedding。

第一层encoder的输入是一个超边内部全部token的表征，输出是这个超边的表征。第二层encoder的输入是的一个token出现的所有超边的表征，输出是这个token更新后的表征。以上的两层算是一次完整的传播。需要进行三次完整传播。


完整的pipeline应该是这样的：
from torch_geometric.data import Data,HeteroData
data = HeteroData()
hyperedges = [[0,1, 2, 3], [4, 5, 6], [2,8,7, 5, 6],[4, 5, 7]] 
hyperedges = [[0,1,2,3,4,5,6,2,8,7,5,6,4,5,7],[0,0,0,0,1,1,1,2,3,3,3,3,4,4,4]]        
data['node', 'in','hyperedge'].edge_index = torch.LongTensor(np.array(hyperedges, dtype = np.int64))

bert_model_name='bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(bert_model_name)
bert = BertModel.from_pretrained(bert_model_name)
data['node'].x = bert.embeddings.word_embeddings(hyperedges[0]) 

class BertConv(MessagePassing):
    def __init__(self, bert_model, in_channels, out_channels):
        self.super().__init__(aggr=None)  # 禁用内置聚合
        self.bert_model = bert_model
        self.lin = torch.nn.Linear(in_channels, out_channels)
        
    def forward(self, x, edge_index):
        # 对节点特征进行预处理，如有必要
        x = self.lin(x)
        # 调用propagate，但不使用内置的聚合函数
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)
        
    def message(self, x_j):
        # 定义消息传递
        return x_j
        
    def aggregate(self, inputs, index, dim_size=None):
        # 使用BERT模型进行聚合
        # 这里需要根据您的具体需求编写代码
        # 例如，将inputs作为BERT输入并获取输出
        bert_output = self.bert_model(inputs)[0]
        return bert_output
        

class HypergraphBERT(torch.nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased'):
        super().__init__()
        bert = BertModel.from_pretrained(bert_model_name)
        
        
        self.layers = torch.nn.ModuleList()
        self.aggr = aggr
        self.bias = bias
        self.num_layers = num_layers
        
        self.dropout = torch.nn.Dropout(dropout)
        self.bn_layers = torch.nn.ModuleList()
        
        n_v = BertConv(bert,in_channels,out_channels)
        v_n = BertConv(bert,in_channels,out_channels)
        
        self.layers.append(HeteroConv({('node','in','hyperedge'): n_v,('hyperedge','rev_in','node'): v_n},aggr = 'sum'))
        self.bn_layers.append(torch.nn.BatchNorm1d(out_channels))
            
            

    def forward(self,data):

        device = data['node'].x.device
        x_dict = {'node':data['node'].x,'hyperedge': torch.zeros((data['hyperedge'].num_nodes, data['node'].x.shape[1])).to(device)}
        edge_index_dict = {
            ('node','in','hyperedge'):data.edge_index_dict['node','in','hyperedge'],
            ('hyperedge','rev_in','node'):reverse_edge(data.edge_index_dict['node','in','hyperedge'])
        }
        
        for i, layer in  enumerate(self.layers):
            x_dict = layer(x_dict,edge_index_dict)

            if i != self.num_layers -1:
                # x_dict = {k:torch.relu(v) for k,v in x_dict.items()}
                x_dict = {k:self.dropout(v) for k,v in x_dict.items()}

        return token_embeddings

bert = HypergraphBERT()
final_token_reprs = bert(data)

给我提供这样的pytorch代码，基于torch_geometric.nn.conv.MessagePassing。我需要能够直接运行的代码。
    
    
    
    
    
    """
    