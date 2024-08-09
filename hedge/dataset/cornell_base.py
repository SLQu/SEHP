from hedge._global import CACHE_ROOT
from .abstract_dataset import AbstractDataset


import os
import os.path as osp
import pickle
import torch
import numpy as np
from torch_geometric.data import Data,HeteroData
from torch_sparse import coalesce
import random


class CornellBase(AbstractDataset):
    
    '''
    
    https://www.cs.cornell.edu/~arb/data/
    
    '''
    
    
    def __init__(self,root = '',
                 hyperedge_size_lowbound = 2,ratio = [0.6, 0.2, 0.2]) -> None:
        
        '''
        root and dataset should be defined before super().__init__(),
        '''
        # citeseer  cora     pubmed
        self.hyperedge_size_lowbound = hyperedge_size_lowbound
        
        super().__init__(root,ratio)
        
        self.data_list = self.processed_file_names
    
    @property
    def raw_dir(self) -> str:
        return osp.join(self.root,self.datasetname)

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, 'processed', self.datasetname,f'hyperedge_size_lowbound_{self.hyperedge_size_lowbound}',self.ratio_str)

  
    def __load__(self):
        nverts = np.loadtxt(osp.join(self.raw_dir, self.raw_file_names[0]),dtype = np.int64)
        simplices = np.loadtxt(osp.join(self.raw_dir, self.raw_file_names[1]),dtype = np.int64)
        times = np.loadtxt(osp.join(self.raw_dir, self.raw_file_names[2]),dtype = np.int64)
        
        with open(osp.join(self.raw_dir, self.raw_file_names[3]), 'r') as file:
            node_labels = [' '.join(line.strip().split()) for line in file]
            
        simplex_labels = None
        return nverts,simplex_labels,simplices,times,node_labels
    
    def align_node_id(self,simplices):
        new_simplices = []
        old_ids = np.unique(simplices)
        new_ids = np.arange(len(old_ids))
        map_dict = dict(zip(old_ids, new_ids))
        
        for n_id in simplices:
            new_simplices.append(map_dict[n_id])
            
        return new_simplices,map_dict
        
    def align_node_label(self,node_labels,map_dict,code= True):

        new_node_labels = [ 0 for i in range(len(map_dict))]
        new_node_codes = [ 0 for i in range(len(map_dict))]
        
        map_dict_keys = map_dict.keys()
        for lb in node_labels:
            lb = lb.split(' ')
            node_id = int(lb[0])
            if node_id not in map_dict_keys:
                continue
            
            if code:
                new_node_labels[map_dict[node_id]] = " ".join(lb[1])
                new_node_codes[map_dict[node_id]] = " ".join(lb[2:])
            else:
                new_node_labels[map_dict[node_id]] = " ".join(lb[1:])
        
        return new_node_labels, new_node_codes
    

    def unzip(self):
        import tarfile
        for __zip__ in self.zip_dir:
            with tarfile.open(__zip__, 'r:gz') as tar:
                tar.extractall(self.raw_dir)
            
    def process(self):
        pass
    
    def format_and_sort_list(self,input_list):
        unique_sorted_list = sorted(set(input_list))
        string_list = [str(item) for item in unique_sorted_list]
        result_string = "-".join(string_list)
        return result_string