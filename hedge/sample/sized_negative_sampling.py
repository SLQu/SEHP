import random
import torch
from typing import Any
from hedge.sample import AbstractNegativeSampling

class SizedNegativeSampling(AbstractNegativeSampling):
    
    '''
    2. Sized Negative Sampling (SNS)  
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
        
        
    def sample(self, data, negative_number = 1000, data_perprocess = False):
        node_ids = []
        hyperedge_ids = []
        edge_index = data['node','in','hyperedge'].edge_index
        num_nodes = data['node'].num_nodes
        num_hyperedges = data['hyperedge'].num_nodes
        
        for edge_id in range(num_hyperedges):
            num_neighbors = (edge_index[1] == edge_id).sum().item()
            if num_neighbors == 0:
                continue

            if data_perprocess and len(hyperedge_ids) > 1:
                if hyperedge_ids[-1][0] == negative_number:
                    break

            for idx in range(self.amount):
                sampled_nodes = self.__single_sample__(num_nodes, num_neighbors)
                node_ids.append(sampled_nodes)
                new_hyperedge_id =  hyperedge_ids[-1][0] + 1 if len(hyperedge_ids) else 0
                hyperedge_ids.append(torch.full((num_neighbors,), new_hyperedge_id, device=self.device))
                
                  
        return torch.cat(node_ids), torch.cat(hyperedge_ids)
        
    def __single_sample__(self,num_nodes, num_neighbors):
        sampled_nodes = torch.randint(0, num_nodes, (num_neighbors,), device=self.device)
        return sampled_nodes
