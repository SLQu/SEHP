import random
import torch
from typing import Any
from hedge.sample import AbstractNegativeSampling
from hedge.utils import format_and_sort_list


    
class MotifNegativeSampling(AbstractNegativeSampling):
    
    '''
    Uniform Negative Sampling
    For a  hypergraph H = (V,F), the UNS algorithm picks a sample of k non-hyperlinks  Fˆsam uniformly at random from the set of all non-hyperlinks Fˆall. The non- hyperlink sizes of Fˆsam are expected to be binomially distributed.
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
                
            
            num_neighbors = sum(edge_index[1] == edge_id)
                
            for idx in range(self.amount):
                
                sampled_nodes = self.__single_sample__(data, num_neighbors)
                if format_and_sort_list(sampled_nodes) in positive_hyperedge_set:
                    continue
                sampled_nodes = torch.tensor(sampled_nodes).to(device=self.device)    
                    
                node_ids.append(sampled_nodes)
                new_hyperedge_id =  hyperedge_ids[-1][0] + 1 if hyperedge_ids else 0
                # print(f'motif:  new_hyperedge_id: {new_hyperedge_id}')
                hyperedge_ids.append((torch.full((sampled_nodes.size()[0],), new_hyperedge_id, device=self.device)))

                  
        return torch.cat(node_ids), torch.cat(hyperedge_ids)
        
    def __single_sample__(self, data, size):

        edge_index = data['node','in','hyperedge'].edge_index
        start_edge = edge_index[:, torch.randint(edge_index.size(1), (1,))]
        sampled_nodes = set(start_edge[0].tolist())
        no_ad_times = 0   

             
        while len(sampled_nodes) < size:
            sampled_nodes_size_last = len(sampled_nodes)
            _sampled_nodes_ = torch.tensor(list(sampled_nodes)).to(device=self.device)
            in_index  = self.is_element_in_tensor(edge_index[0],_sampled_nodes_)    
            connected_edges = edge_index[1, in_index]
            in_index = self.is_element_in_tensor(edge_index[1], connected_edges)
            shared_nodes = edge_index[0, in_index]
            chosen_nodes = shared_nodes[torch.randint(shared_nodes.size(0), (1,))]
            sampled_nodes.add(chosen_nodes.item())
            if len(sampled_nodes) == sampled_nodes_size_last:
                no_ad_times += 1
            
            if no_ad_times == 5:
                    break

        return list(sampled_nodes)
    
