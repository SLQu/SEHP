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


class DAWN(CornellBase):
    
    '''
    
    https://www.cs.cornell.edu/~arb/data/DAWN/
    
    
    original dataset:
    number of nodes: 2,558
    number of timestamped simplices: 2,272,433
    number of unique simplices: 143,523
    number of edges in projected graph: 122,963
    
    real dataset:
    number of nodes: 2558
    number of unique simplices: 138,742
    number of edge_idex_num: 553,159
    
    
    '''
    
    
    def __init__(self, datasetname = 'DAWN',root = '',
                 hyperedge_size_lowbound = 2,ratio = [0.6, 0.2, 0.2]) -> None:
        
        '''
        root and dataset should be defined before super().__init__(),
        '''
        # citeseer  cora     pubmed
        self.datasetname = 'DAWN' 
        root = CACHE_ROOT if root == '' else root
        
        super().__init__(root, hyperedge_size_lowbound,ratio)
        
        self.data_list = self.processed_file_names

    @property
    def raw_file_names(self):
        raw = ['DAWN-nverts.txt','DAWN-simplices.txt','DAWN-times.txt','DAWN-node-labels.txt']
        
        raw = [ osp.join('DAWN',name) for name in raw] 
        return raw
    
    
    def __load__(self):

        nverts = np.loadtxt(osp.join(self.raw_dir, self.raw_file_names[0]),dtype = np.int64)
        simplices = np.loadtxt(osp.join(self.raw_dir, self.raw_file_names[1]),dtype = np.int64)
        times = np.loadtxt(osp.join(self.raw_dir, self.raw_file_names[2]),dtype = np.int64)
        
        with open(osp.join(self.raw_dir, self.raw_file_names[3]), 'r') as file:
            node_labels = [' '.join(line.strip().split()) for line in file]

        simplex_labels = None
        return nverts,simplex_labels,simplices,times,node_labels
    
   
    @property
    def zip_dir(self):
        
        return [osp.join(self.raw_dir,'zip','DAWN-proj-graph.tar.gz'),
                osp.join(self.raw_dir,'zip','DAWN.tar.gz')]

            
    def process(self):
        print(f'root: {self.root}')
        print('processing... begin')

        self.unzip()
        
        nverts,simplex_labels,simplices,times,node_labels = self.__load__()
        
        simplices,  map_dict = self.align_node_id(simplices)
        node_labels, node_codes = self.align_node_label(node_labels,map_dict, code = True)
        
        
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
            # new_simplex_labels.append(simplex_labels[idx])
            new_times.append(times[idx])
        
 
        ### remap node id.
        node_ids = dict()
        for i, edge in enumerate(new_hedge):
            _temp_ = set()
            for n_id in edge:
                if n_id not in node_ids:
                    node_ids[n_id] = len(node_ids)
                _temp_.add(node_ids[n_id])
            new_hedge[i] = _temp_

        idx_map = [ -1 for i in range(len(node_ids))]

        for k in node_ids.keys():
            idx_map[node_ids[k]] = k

        node_labels = [node_labels[i] for i in idx_map]

            # labels = labels[idx_map]
            # features = features[idx_map]


        ################
               
        data = HeteroData()
        hyperedge_ids = []
        node_ids = []
        for idx, nodes in enumerate(new_hedge):
            hyperedge_ids += [idx] * len(nodes)
            node_ids += nodes
            
        data['node', 'in','hyperedge'].edge_index = torch.LongTensor(np.array([node_ids, hyperedge_ids], dtype = np.int64))
        # data['hyperedge', 'node'].edge_index = torch.LongTensor(np.array([hyperedge_ids, node_ids], dtype = np.int64))
        data['hyperedge'].num_nodes = len(new_hedge)
        data['node', 'in','hyperedge'].num_edge_index = len(node_ids)
        # data['hyperedge'].labels = new_simplex_labels  
        data['hyperedge'].timestamp = torch.LongTensor(new_times)
        data['node'].num_nodes = len(node_labels)
        # data['node'].node_labels_txt = node_labels
        
        # data['node'].x = torch.nn.Embedding(num_embeddings = data['node'].num_nodes, embedding_dim = 333)
        data['node'].x = torch.eye(data['node'].num_nodes)  




        #######################################################
        print(f'begin spliting hyperedges')
        data = self.hyperedge_split(data, mode = 'random')
        print(f'end spliting hyperedges')
        data = self.set_negative(data)
        #######################################################
        torch.save(data, osp.join(self.processed_dir, self.processed_file_names[0]))
        print('processing... end')
            
            
        



