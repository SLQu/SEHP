from hedge._global import CACHE_ROOT
from .cornell_base import CornellBase
from hedge.utils import format_and_sort_list

import os
import os.path as osp
import pickle
import torch
import numpy as np
from torch_geometric.data import Data,HeteroData
from torch_sparse import coalesce
import random


class NDC_substances(CornellBase):
    
    '''
    
    https://www.cs.cornell.edu/~arb/data/NDC-substances/
    
    original dataset:
    number of nodes: 5,311
    number of timestamped simplices: 112,405
    number of unique simplices: 10,025
    number of edges in projected graph: 88,268
    
    real dataset:
    number of nodes: 5311
    number of unique simplices: 6,264
    number of edge_idex_num: 49,886
    
    
    '''
    def __init__(self, datasetname = 'NDC_substances',root = '', hyperedge_size_lowbound = 2,ratio = [0.6, 0.2, 0.2]) -> None:
        
        '''
        root and dataset should be defined before super().__init__(),
        '''
        # citeseer  cora     pubmed
        root = CACHE_ROOT if root == '' else root
        self.datasetname = 'NDC_substances_25' if 'full' not in datasetname else 'NDC_substances_full'
        
        
        super().__init__(root = root,hyperedge_size_lowbound = hyperedge_size_lowbound,ratio = ratio)
        
        self.data_list = self.processed_file_names
        
    @property
    def raw_dir(self) -> str:
        return osp.join(self.root,self.datasetname,'NDC-substances')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, 'processed','NDC_substances', self.datasetname,f'hyperedge_size_lowbound_{self.hyperedge_size_lowbound}',self.ratio_str)

    @property
    def raw_file_names(self):
        raw = ['NDC-substances-nverts.txt','NDC-substances-simplex-labels.txt','NDC-substances-simplices.txt','NDC-substances-times.txt','NDC-substances-node-labels.txt']
        raw = [ osp.join('NDC-substances',name) for name in raw] 
        return raw
    
    def __load__(self):

        nverts = np.loadtxt(osp.join(self.raw_dir, self.raw_file_names[0]),dtype = np.int64)

        with open(osp.join(self.raw_dir, self.raw_file_names[1]), 'r') as file:
            simplex_labels = [line.strip() for line in file]
            
        simplices = np.loadtxt(osp.join(self.raw_dir, self.raw_file_names[2]),dtype = np.int64)
        times = np.loadtxt(osp.join(self.raw_dir, self.raw_file_names[3]),dtype = np.int64)
        
        with open(osp.join(self.raw_dir, self.raw_file_names[4]), 'r') as file:
            node_labels = [' '.join(line.strip().split()) for line in file]

        return nverts,simplex_labels,simplices,times,node_labels
    
    
    @property
    def zip_dir(self):
        
        return [osp.join(self.source_dir,'zip','NDC-substances-proj-graph.tar.gz'),
                osp.join(self.source_dir,'zip','NDC-substances.tar.gz')]
    @property
    def source_dir(self) -> str:
        return self.raw_dir.strip('NDC-substances')
            
    def process(self):
        print(f'root: {self.root}')
        print('processing... begin')
        self.unzip()
        
        nverts,simplex_labels,simplices,times,node_labels = self.__load__()
        simplices,  map_dict = self.align_node_id(simplices)
        node_labels, node_codes = self.align_node_label(node_labels,map_dict, code = False)
        
        
        raw_hedge = []
        start_idx = 0
        for n in nverts.tolist():
            end_idx = start_idx + n
            simplex = simplices[start_idx:end_idx]
            raw_hedge.append(simplex)
            start_idx = end_idx 
        
        target_hedge = dict()
        


        for idx,e in enumerate(raw_hedge):
            _key_ = format_and_sort_list(e)
            if _key_ not in target_hedge.keys():
                target_hedge[_key_] = (idx,e)
                
        print(f'number of target hyperedges: {len(target_hedge)}')
        
        new_simplex_labels = []
        new_times = []
        new_hedge = []
        
        for _key_ in target_hedge.keys():
            idx,e = target_hedge[_key_]
            if  len(e) < self.hyperedge_size_lowbound:
                continue
            new_hedge.append(e)
            
            new_simplex_labels.append(simplex_labels[idx])
            new_times.append(times[idx])
        
    
        data = HeteroData()
        hyperedge_ids = []
        node_ids = []
        for idx, nodes in enumerate(new_hedge):
            hyperedge_ids += [idx] * len(nodes)
            node_ids += nodes
            
        data['node', 'in','hyperedge'].edge_index = torch.LongTensor(np.array([node_ids, hyperedge_ids], dtype = np.int64))
        data['node', 'in','hyperedge'].num_edge_index = len(node_ids)
        data['hyperedge'].num_nodes = len(new_hedge)
        data['hyperedge'].timestamp = torch.LongTensor(new_times)
        data['node'].num_nodes = len(node_labels)
        data['node'].x = torch.randn(data['node'].num_nodes, 333)

 
        #######################################################
        print(f'begin spliting hyperedges')
        data = self.hyperedge_split(data, mode = 'random')
        print(f'end spliting hyperedges')
        data = self.set_negative(data)
        #######################################################
        torch.save(data, osp.join(self.processed_dir, self.processed_file_names[0]))
        print('processing... end')
            
            
   
   
class NDC_substances_full(NDC_substances):
    def __init__(self, datasetname = 'NDC_substances_full',root = '', hyperedge_size_lowbound = 2,ratio = [0.6, 0.2, 0.2]) -> None:
        
        '''
        root and dataset should be defined before super().__init__(),
        '''
        # citeseer  cora     pubmed
        root = CACHE_ROOT if root == '' else root
        
        super().__init__(root = root,hyperedge_size_lowbound = hyperedge_size_lowbound,ratio = ratio)
        
        
        self.data_list = self.processed_file_names  
            
            
    @property
    def raw_dir(self) -> str:
        return osp.join(self.root,self.datasetname,'NDC-substances-full')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, 'processed','NDC_substances', self.datasetname,f'hyperedge_size_lowbound_{self.hyperedge_size_lowbound}',self.ratio_str)

    @property
    def raw_file_names(self):
        raw = ['NDC-substances-full-nverts.txt','NDC-substances-full-simplex-labels.txt','NDC-substances-full-simplices.txt','NDC-substances-full-times.txt','NDC-substances-full-node-labels.txt']
        raw = [ osp.join('NDC-substances-full',name) for name in raw] 
        return raw
    
    
    @property
    def zip_dir(self):
        return [osp.join(self.source_dir,'zip','NDC-substances-full-proj-graph.tar.gz'),
                osp.join(self.source_dir,'zip','NDC-substances-full.tar.gz')]
    @property
    def source_dir(self) -> str:
        return self.raw_dir.strip('NDC-substances-full')


