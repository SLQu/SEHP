import random
import torch
from typing import Any
from hedge.sample import AbstractNegativeSampling
from hedge.utils import format_and_sort_list
class EdgeRepDiffusionGeneratedNegativeSampling(AbstractNegativeSampling):
    
    '''
     diffusion Generated Negative Sampling (GNS)  
    
    '''
    
    def __init__(self, aggregator, amount: int = 1, device = torch.device('cpu'), encoder = None, decoder = None, stru_diff = 1, stru_decoder = 0, way_inject_edge_rep  = 'concatenate', share_encoder = False, diffusion_step = 1) -> None:
        super().__init__(amount, device)
        
        self.encoder = encoder
        self.decoder = decoder[0]
        self.alignment_layer = decoder[1]

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
        self.alignment_layer.eval()
    
    def train(self):
        self.encoder.train()
        self.decoder.train()
        self.alignment_layer.train()


    def sample(self, data, positive_hyperedge_emb = None, negative_number = 1000, data_perprocess = False):

        num_nodes = data['node'].num_nodes
        num_hyperedges = data['hyperedge'].num_nodes

        hedge_reps = []

        x_dict = data['x_dict'] if self.share_encoder else self.encoder(data)
        node_rep = x_dict['node']

        hedge_representation = positive_hyperedge_emb if positive_hyperedge_emb is not None else torch.randn(num_hyperedges, node_rep.shape[1], device=self.device)

        graph_structure = self.aggregator(node_rep,torch.zeros(num_nodes, dtype=torch.long, device=self.device))
        graph_structure = graph_structure.expand(num_hyperedges, -1)


        for i in range(self.diffusion_step):
            noise = torch.randn(num_hyperedges, node_rep.shape[1], device=self.device)
            input_of_decoder = torch.cat([graph_structure + noise, hedge_representation], dim = 1) if self.stru_diff else torch.cat([noise, hedge_representation], dim = 1)

            neg_hedge_representation = self.decoder(input_of_decoder)  

            if i > self.diffusion_step - self.amount - 1:
                hedge_reps.append(neg_hedge_representation)

        hedge_reps_all = torch.cat(hedge_reps)
        
        return hedge_reps_all
        # return self.alignment_layer(hedge_reps_all)
    
