import random
import torch
from typing import Any
from hedge.sample import AbstractNegativeSampling
from hedge.utils import format_and_sort_list
class PlusGeneratedNegativeSampling(AbstractNegativeSampling):
    
    '''
    2. Generated Negative Sampling (GNS)  
    
    
    
    '''
    
    def __init__(self, amount: int = 1, device = torch.device('cpu'), encoder = None, decoder = None) -> None:
        super().__init__(amount, device)
        
        self.encoder = encoder
        self.decoder = decoder
        
        
    def sample(self, data, negative_number = 1000, data_perprocess = False):
        
        node_ids = []
        hyperedge_ids = []
        edge_index = data['node','in','hyperedge'].edge_index
        num_nodes = data['node'].num_nodes
        print(f'num_nodes,{num_nodes}')
        num_hyperedges = data['hyperedge'].num_nodes
        
        positive_hyperedge_set = self.positive_base(data)
        
        x_dict = self.encoder(data)
        node_rep = x_dict['node']
        device = node_rep.device


        for edge_id in range(num_hyperedges):
            num_neighbors = (edge_index[1] == edge_id).sum().item()
            if data_perprocess and len(hyperedge_ids) > 1:
                if hyperedge_ids[-1][0] == negative_number:
                    break
            
            for idx in range(self.amount):
                
                sampled_nodes = self.__single_sample__(node_rep,num_nodes , num_neighbors)
                
                if format_and_sort_list(sampled_nodes.to('cpu').tolist()) in positive_hyperedge_set:
                    continue
                
                # sampled_nodes = torch.tensor(sampled_nodes).to(device=self.device) 
                node_ids.append(sampled_nodes)
    
                new_hyperedge_id =  hyperedge_ids[-1][0] + 1 if hyperedge_ids else 0
                hyperedge_ids.append(torch.full((num_neighbors,), new_hyperedge_id, device=self.device))
                  
        return torch.cat(node_ids), torch.cat(hyperedge_ids)
    
    def eval(self):
        self.encoder.eval()
        self.decoder.eval()
    
    def train(self):
        self.encoder.train()
        self.decoder.train()
        
    def __single_sample__(self,node_rep,num_nodes, num_neighbors):
        
        variance = 1
        std_dev = variance ** 0.5  
        
        
        base = torch.zeros(1, num_nodes, device=self.device) 
        node_rep = node_rep.view(1,-1)
        while True:
            noise = torch.randn(1, num_nodes, device=self.device)  * std_dev
            # input = torch.cat((node_rep + noise, base), dim = 1) 
            
            input = node_rep + noise
            
            all_possibility = self.decoder(input)                
            values, sampled_nodes = torch.topk(all_possibility.squeeze(), k=num_neighbors)
            
            if max(sampled_nodes) < num_nodes:
                break

        return sampled_nodes
