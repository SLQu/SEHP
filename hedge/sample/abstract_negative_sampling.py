import random
import torch
from typing import Any

from hedge.utils import convert_bipartite_to_list_hyperedges, format_and_sort_list

class AbstractNegativeSampling():
    def __init__(self, amount: int = 1, device = torch.device('cpu'), negative_number = 1000) -> None:
        
        '''
        amount is the number of negative samples per hyperedge
        '''
        self.amount = amount
        self.device = device 
        self.negative_hyperedge_id = 0
        # self.negative_number = negative_number

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.sample(*args, **kwds)
    
    def sample(self, data):
        pass
    
    def get_negative_hyperedge_id(self):
        self.negative_hyperedge_id += 1
        return self.negative_hyperedge_id - 1
    
    def set_amount(self, amount: int):
        
        # write code to check if amount is digit or not
        if isinstance(amount, int) == False:
            raise ValueError("amount must be integer")
        
        if amount < 1:
            raise ValueError("amount must be at least 1")
        
        self.amount = amount
    
    def positive_base(self, data):
        hyperedge_list = convert_bipartite_to_list_hyperedges(data['node','in','hyperedge'].edge_index)
        
        hyperedge_set = set()
        for e in hyperedge_list:
            hyperedge_set.add(format_and_sort_list(e)) 
        
        return hyperedge_set
    
    def is_element_in_tensor(self,t1, t2):
        '''
        
        t1 = tensor([ 0,  1 ,0,3,2,1,2,4,5,6,3]
        t2 = tensor([ 0,3]
        
        return tensor([ true,  false,true,true,false,false,false,false,false,false,true]
        '''
        
        t2_tensor = t2.unsqueeze(0)
        t1_expanded = t1.unsqueeze(1)
        result = torch.any(t1_expanded == t2_tensor, dim=1)
        return result
    