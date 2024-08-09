import random
import torch
from typing import Any
from hedge.sample import AbstractNegativeSampling
from hedge.utils import format_and_sort_list
class RepDiffusionGeneratedNegativeSampling(AbstractNegativeSampling):
    
    '''
     diffusion Generated Negative Sampling (GNS)  
    
    '''
    
    def __init__(self, aggregator, amount: int = 1, device = torch.device('cpu'), encoder = None, decoder = None, stru_diff = 1,stru_decoder = 1, way_inject_edge_rep  = 'concatenate', share_encoder = False, diffusion_step = 1) -> None:
        super().__init__(amount, device)
        
        self.encoder = encoder
        self.decoder = decoder[0]
        self.decoder_node_ids = decoder[1]
        self.aggregator = aggregator
        self.way_inject_edge_rep  = way_inject_edge_rep
        self.share_encoder = share_encoder
        self.diffusion_step = diffusion_step

        if stru_diff == 1:
            self.stru_diff = True
        else:
            self.stru_diff = False
        

        if stru_decoder == 1:
            self.stru_decoder = True
        else:
            self.stru_decoder = False
        
    def eval(self):
        self.encoder.eval()
        self.decoder.eval()
        self.decoder_node_ids.eval()
    
    def train(self):
        self.encoder.train()
        self.decoder.train()
        self.decoder_node_ids.train()


    def sample(self, data, positive_hyperedge_emb = None, negative_number = 1000, data_perprocess = False, analysis_step = False):
        node_ids = []
        hyperedge_ids = []
        edge_index = data['node','in','hyperedge'].edge_index
        num_nodes = data['node'].num_nodes
        num_hyperedges = data['hyperedge'].num_nodes
        positive_hyperedge_set = self.positive_base(data)
        hedge_reps = []
        
        if self.share_encoder:
            x_dict = data['x_dict']
        else:
            x_dict = self.encoder(data)

        node_rep = x_dict['node']
        device = node_rep.device
        if positive_hyperedge_emb is not None:
            hedge_representation = positive_hyperedge_emb
        else:
            hedge_representation = torch.randn(num_hyperedges, node_rep.shape[1], device=self.device)
            
        graph_structure = self.aggregator(node_rep,torch.zeros(num_nodes, dtype=torch.long, device=self.device))
        graph_structure = graph_structure.expand(num_hyperedges, -1)

        for i in range(self.diffusion_step):
            noise = torch.randn(num_hyperedges, node_rep.shape[1], device=self.device)

            if self.stru_diff:
                input_of_decoder = torch.cat([graph_structure + noise, hedge_representation], dim = 1)
            else:
                input_of_decoder = torch.cat([noise, hedge_representation], dim = 1)

            hedge_representation = self.decoder(input_of_decoder)  

            if analysis_step:
                for edge_id in range(num_hyperedges):
                        num_neighbors = (edge_index[1] == edge_id).sum().item()
                        if num_neighbors == 0:
                            continue

                        sampled_nodes, single_hedge_representation = self.__single_sample__(node_rep,num_nodes , num_neighbors, hedge_rep = hedge_representation[edge_id])
                        while format_and_sort_list(sampled_nodes.to('cpu').tolist()) in positive_hyperedge_set:
                            sampled_nodes, single_hedge_representation = self.__single_sample__(node_rep,num_nodes , num_neighbors, hedge_rep = hedge_representation[edge_id])
                            
                        node_ids.append(sampled_nodes)
                        new_hyperedge_id =  hyperedge_ids[-1][0] + 1 if hyperedge_ids else 0
                        hyperedge_ids.append(torch.full((num_neighbors,), new_hyperedge_id, device=self.device))
                        hedge_reps.append(single_hedge_representation)
            else:

                if i > self.diffusion_step - self.amount - 1:

                    for edge_id in range(num_hyperedges):
                        num_neighbors = (edge_index[1] == edge_id).sum().item()
                        if num_neighbors == 0:
                            continue

                        sampled_nodes, single_hedge_representation = self.__single_sample__(node_rep,num_nodes , num_neighbors, hedge_rep = hedge_representation[edge_id])
                        while format_and_sort_list(sampled_nodes.to('cpu').tolist()) in positive_hyperedge_set:
                            sampled_nodes, single_hedge_representation = self.__single_sample__(node_rep,num_nodes , num_neighbors, hedge_rep = hedge_representation[edge_id])
                            
                        node_ids.append(sampled_nodes)
                        new_hyperedge_id =  hyperedge_ids[-1][0] + 1 if hyperedge_ids else 0
                        hyperedge_ids.append(torch.full((num_neighbors,), new_hyperedge_id, device=self.device))
                        hedge_reps.append(single_hedge_representation)


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
            
            if self.stru_decoder:
                input_of_decoder = torch.cat([graph_structure + noise, hedge_representation], dim = 1)
            else:
                input_of_decoder = torch.cat([noise, hedge_representation], dim = 1)

        else:
            raise NotImplementedError

        all_possibility = self.decoder_node_ids(input_of_decoder)  

        values, sampled_nodes = torch.topk(all_possibility.squeeze(), k=num_neighbors)
        hedge_representation =  self.aggregator(node_rep[sampled_nodes], torch.zeros(num_neighbors, dtype=torch.long, device=self.device))

        return sampled_nodes, hedge_representation
