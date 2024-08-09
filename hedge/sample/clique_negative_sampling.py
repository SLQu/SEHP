import random
import torch
from typing import Any
from hedge.sample import AbstractNegativeSampling

from hedge.utils import format_and_sort_list

    
class CliqueNegativeSampling(AbstractNegativeSampling):
    
    
    '''

    
    '''
    
    def __init__(self, amount: int = 1, device = torch.device('cpu')) -> None:
        super().__init__(amount, device)

    
    def sample(self, data, negative_number = 1000, data_perprocess = False):
        node_ids = []
        hyperedge_ids = []
        edge_index = data['node','in','hyperedge'].edge_index
        num_nodes = data['node'].num_nodes
        
        positive_hyperedge_set = self.positive_base(data)
        
        
        for edge_id in range(data['hyperedge'].num_nodes):
            
            if data_perprocess and len(hyperedge_ids) > 1:
                if hyperedge_ids[-1][0] == negative_number:
                    break
                
            for idx in range(self.amount):
                sampled_nodes = self.__single_sample__(data)
                
                if format_and_sort_list(sampled_nodes) in positive_hyperedge_set:
                    continue
                sampled_nodes = torch.tensor(sampled_nodes).to(device=self.device)    
                
    
                node_ids.append(sampled_nodes)
                new_hyperedge_id =  hyperedge_ids[-1][0] + 1 if hyperedge_ids else 0
                # print(f'clique:  new_hyperedge_id: {new_hyperedge_id}')
                
                hyperedge_ids.append(torch.full((sampled_nodes.size()[0],), new_hyperedge_id, device=self.device))

                  
        return torch.cat(node_ids), torch.cat(hyperedge_ids)
        
    def __single_sample__(self, data):
                
        edge_index = data['node','in','hyperedge'].edge_index

        chosen_edge_id = random.randint(0, data['hyperedge'].num_nodes - 1)
        chosen_edge = edge_index[0, edge_index[1] == chosen_edge_id]

        removed_node = random.choice(chosen_edge.tolist())
        new_edge = chosen_edge[chosen_edge != removed_node].tolist()

        all_nodes = set(range(data['node'].num_nodes))
        available_nodes = list(all_nodes - set(new_edge))
        new_node = random.choice(available_nodes)
        new_edge.append(new_node)

        return new_edge
    

    
    
