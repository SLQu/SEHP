import random
import torch
from typing import Any
from hedge.sample import AbstractNegativeSampling

  

class UniformNegativeSampling(AbstractNegativeSampling):
    
    '''
    1. Uniform Negative Sampling (UNS)  
    SNS is a slight variant of UNS in that  the target SD 
    (i.e., that of the sampled negative class) P r−(S = s) 
    is fixed  to that of the positive class (P r+(S = s)), 
    and not a binomial. 
    Once a size s has been sampled according to P r+(S = s), 
    a non-hyperlink is  sampled randomly. 
    The SD of non-hyperlinks sampled with SNS exactly follows  
    the positive class SD.    
    Size distribution (SD)
    '''
    
    def __init__(self, amount: int = 1, device = torch.device('cpu')) -> None:
        super().__init__(amount, device)
        
        
    def sample(self, data):
        node_ids = []
        hyperedge_ids = []
        edge_index = data['node','in','hyperedge'].edge_index
        num_nodes = data['node'].num_nodes
        
        for edge_id in range(data['hyperedge'].num_nodes):
            num_neighbors = sum(edge_index[1] == edge_id)
            new_hyperedge_id_base = edge_id * self.amount
            for idx in range(self.amount):
                node_ids += random.sample(range(0, num_nodes), num_neighbors)
                hyperedge_ids += [new_hyperedge_id_base + idx] * num_neighbors
                
        return (torch.LongTensor(node_ids).to(self.device), torch.LongTensor(hyperedge_ids).to(self.device)) 
        
