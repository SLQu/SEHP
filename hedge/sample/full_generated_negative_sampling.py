import random
import torch
from typing import Any
from hedge.sample import AbstractNegativeSampling
from hedge.utils import format_and_sort_list
class FullGeneratedNegativeSampling(AbstractNegativeSampling):
    
    '''
    2. Generated Negative Sampling (GNS)  
    
    
    
    '''
    
    def __init__(self, amount: int = 1, device = torch.device('cpu'), decoder = None) -> None:
        super().__init__(amount, device)
        
        self.decoder = decoder
        
        
    def sample(self, data, negative_number = 1000, data_perprocess = False):
        
        node_ids = []
        hyperedge_ids = []
        edge_index = data['node','in','hyperedge'].edge_index
        num_nodes = data['node'].num_nodes
        num_hyperedges = data['hyperedge'].num_nodes
        
        positive_hyperedge_set = self.positive_base(data)
        # print(f'{list(positive_hyperedge_set)[:10]}')
        

        for edge_id in range(num_hyperedges):
            num_neighbors = (edge_index[1] == edge_id).sum().item()
            if data_perprocess and len(hyperedge_ids) > 1:
                if hyperedge_ids[-1][0] == negative_number:
                    break
            
            for idx in range(self.amount):
                
                
                sampled_nodes = self.__single_sample__(num_neighbors)
                
                if format_and_sort_list(sampled_nodes.to('cpu').tolist()) in positive_hyperedge_set:
                    continue
                
                # sampled_nodes = torch.tensor(sampled_nodes).to(device=self.device) 
                node_ids.append(sampled_nodes)
    
                new_hyperedge_id =  hyperedge_ids[-1][0] + 1 if hyperedge_ids else 0
                hyperedge_ids.append(torch.full((num_neighbors,), new_hyperedge_id, device=self.device))
                
                  
        return torch.cat(node_ids), torch.cat(hyperedge_ids)
    
    def __single_sample__(self, num_neighbors):
        
        variance = 1
        std_dev = variance ** 0.5  
        
        noise = torch.randn(1, 64, device=self.device)  * std_dev
        all_possibility = self.decoder(noise)                
        values, sampled_nodes = torch.topk(all_possibility.squeeze(), k=num_neighbors)

        return sampled_nodes
    
    
    def eval(self):
        self.decoder.eval()
    
    def train(self):
        self.decoder.train()
        

