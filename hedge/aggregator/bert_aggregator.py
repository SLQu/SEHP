
from typing import Optional
from torch_geometric.nn.aggr import Aggregation
from torch import Tensor
import torch

class BertAggregation(Aggregation):
    def __init__(
        self,
        nlp_model
    ):
        super().__init__()

        self.encoder = nlp_model.encoder

    def forward(self, x: Tensor, index: Optional[Tensor] = None,
                ptr: Optional[Tensor] = None, dim_size: Optional[int] = None,
                dim: int = -2) -> Tensor:
                
        group_sizes = torch.bincount(index)
        max_size = torch.max(group_sizes)
        new_x = torch.zeros(len(group_sizes), max_size, x.size(1), device=x.device)

        start_idx = 0
        for i, size in enumerate(group_sizes):
            new_x[i, :size] = x[start_idx:start_idx + size]
            start_idx += size
            
        x = new_x
        with torch.no_grad():
            _output_ = self.encoder(x).last_hidden_state.mean(dim=1)        
        return _output_
    
    
    
'''
ssh sx

cd bi 

'''
    