import random
import torch
from typing import Any
from hedge.sample import AbstractNegativeSampling
from hedge.utils import format_and_sort_list
class DiffusionGeneratedNegativeSampling(AbstractNegativeSampling):
    
    '''
     diffusion Generated Negative Sampling (GNS)  
    
    '''
    
    def __init__(self, aggregator, amount: int = 1, device = torch.device('cpu'), encoder = None, decoder = None, stru_diff = 1, stru_decoder = 0, way_inject_edge_rep  = 'concatenate', share_encoder = False, diffusion_step = 1) -> None:
        super().__init__(amount, device)
        
        self.encoder = encoder
        self.decoder = decoder
        self.aggregator = aggregator
        self.way_inject_edge_rep  = way_inject_edge_rep
        self.share_encoder = share_encoder
        self.diffusion_step = diffusion_step

        if stru_diff == 1:
            self.stru_diff = True
        else:
            self.stru_diff = False
        
    def eval(self):
        self.encoder.eval()
        self.decoder.eval()
    
    def train(self):
        self.encoder.train()
        self.decoder.train()


    def sample(self, data, negative_number = 1000, data_perprocess = False):
        node_ids = []
        hyperedge_ids = []
        edge_index = data['node','in','hyperedge'].edge_index
        num_nodes = data['node'].num_nodes
        num_hyperedges = data['hyperedge'].num_nodes
        positive_hyperedge_set = self.positive_base(data)
        hedge_reps = []

        if self.diffusion_step < self.amount:
            print('diffusion step should be larger than amount')
            print(f"diffusion step: {self.diffusion_step}, amount: {self.amount}")
            exit()
        
        if self.share_encoder:
            x_dict = data['x_dict']
        else:
            x_dict = self.encoder(data)

        node_rep = x_dict['node']
        device = node_rep.device

        for edge_id in range(num_hyperedges):
            num_neighbors = (edge_index[1] == edge_id).sum().item()
            

            _, hedge_representation = self.__single_sample__(node_rep,num_nodes , num_neighbors, hedge_rep = None)

            idx = 0
            for i in range(self.diffusion_step):
                if not idx < self.amount:
                    break
                
                sampled_nodes, hedge_representation = self.__single_sample__(node_rep,num_nodes , num_neighbors, hedge_rep = hedge_representation)

                if i > self.diffusion_step - self.amount -1 :
                    while format_and_sort_list(sampled_nodes.to('cpu').tolist()) in positive_hyperedge_set:
                        sampled_nodes, hedge_representation = self.__single_sample__(node_rep,num_nodes , num_neighbors, hedge_rep = hedge_representation)
                        
                    node_ids.append(sampled_nodes)
                    new_hyperedge_id =  hyperedge_ids[-1][0] + 1 if hyperedge_ids else 0
                    hyperedge_ids.append(torch.full((num_neighbors,), new_hyperedge_id, device=self.device))
                    hedge_reps.append(hedge_representation)
                    idx += 1

            # while idx < self.amount:
            #     if idx == 0:
            #         sampled_nodes, hedge_representation = self.__single_sample__(node_rep,num_nodes , num_neighbors, hedge_rep = None)
            #     else:
            #         for i in range(self.diffusion_step):
            #             sampled_nodes, hedge_representation = self.__single_sample__(node_rep,num_nodes , num_neighbors, hedge_rep = hedge_representation)

            #     if format_and_sort_list(sampled_nodes.to('cpu').tolist()) in positive_hyperedge_set:
            #         continue
                
            #     node_ids.append(sampled_nodes)
            #     new_hyperedge_id =  hyperedge_ids[-1][0] + 1 if hyperedge_ids else 0
            #     hyperedge_ids.append(torch.full((num_neighbors,), new_hyperedge_id, device=self.device))
            #     hedge_reps.append(hedge_representation)
            #     idx += 1

        return torch.cat(node_ids), torch.cat(hyperedge_ids), torch.cat(hedge_reps)
    

        
    def __single_sample__(self,node_rep,num_nodes, num_neighbors, hedge_rep = None):
        
        
        variance = 1
        std_dev = variance ** 0.5  
        graph_structure = node_rep

        if hedge_rep is not None:
            hedge_representation = hedge_rep.expand(num_nodes, -1)
        else:
            hedge_representation = torch.randn(num_nodes, node_rep.shape[1], device=self.device)  * std_dev

        if self.way_inject_edge_rep == 'concatenate':
            noise = torch.randn(num_nodes, node_rep.shape[1], device=self.device)  * std_dev
            
            if self.stru_diff:
                input_of_decoder = torch.cat([graph_structure + noise, hedge_representation], dim = 1)
            else:
                input_of_decoder = torch.cat([noise, hedge_representation], dim = 1)

        else:
            raise NotImplementedError

        all_possibility = self.decoder(input_of_decoder)  

        values, sampled_nodes = torch.topk(all_possibility.squeeze(), k=num_neighbors)
        hedge_representation =  self.aggregator(node_rep[sampled_nodes], torch.zeros(num_neighbors, dtype=torch.long, device=self.device))

        return sampled_nodes, hedge_representation
