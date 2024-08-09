from hedge._global import CACHE_ROOT
from .cornell_base import AbstractDataset
from hedge.utils import format_and_sort_list

import os
import os.path as osp
import pickle
import torch
import numpy as np
from torch_geometric.data import Data,HeteroData
from torch_sparse import coalesce
import random


class Cooking(AbstractDataset):
    
    '''
    
    https://www.cs.cornell.edu/~arb/data/cat-edge-Cooking/
    Hypergraph where nodes are food ingredients, hyperedges are recipes made from 
    combining multiple ingredients and categories indicate cuisine
    (e.g., "Southern-US", "Indian", "Spanish"). 

    Raw data was obtained from the "What's Cooking?" Kaggle competition. Some summary statistics of the network are:
    number of nodes: 6,714
    number of hyperedges: 39,774
    number of edge label categories: 20
    rank of hypergraph (maximum hyperedge size): 65
    '''
    
    
    def __init__(self, datasetname = 'cooking',root = '',
                 hyperedge_size_lowbound = 2,ratio = [0.6, 0.2, 0.2]) -> None:
        
        '''
        root and dataset should be defined before super().__init__(),
        '''
        # citeseer  cora     pubmed
        self.datasetname = datasetname
        root = CACHE_ROOT if root == '' else root
        self.hyperedge_size_lowbound = hyperedge_size_lowbound
        super().__init__(root,ratio)
        
        self.data_list = self.processed_file_names

    @property
    def raw_file_names(self):
        raw = ['hyperedge-label-identities.txt','hyperedges.txt','hyperedge-labels.txt','node-labels.txt']
        
        raw = [ osp.join('cat-edge-Cooking',name) for name in raw] 
        return raw
    
    @property
    def raw_dir(self) -> str:
        return osp.join(self.root,self.datasetname)

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, 'processed', self.datasetname,f'hyperedge_size_lowbound_{self.hyperedge_size_lowbound}',self.ratio_str)
    def __load__(self):

        
        with open(osp.join(self.raw_dir, self.raw_file_names[0]), 'r') as file:
            hyperedge_label_identities = [line.strip() for line in file]
            
        with open(osp.join(self.raw_dir, self.raw_file_names[1]), 'r') as file:
            hyperedges = [line.strip().split('	') for line in file]
        
        for i,e in enumerate(hyperedges):
            hyperedges[i] = [int(j)-1  for j in e]
        
        hyperedge_labels = np.loadtxt(osp.join(self.raw_dir, self.raw_file_names[2]),dtype = np.int64)
        
        with open(osp.join(self.raw_dir, self.raw_file_names[3]), 'r') as file:
            node_labels = [line.strip().split() for line in file]

        return hyperedge_label_identities,hyperedges,hyperedge_labels,node_labels
    
   
    @property
    def zip_dir(self):
        
        return [osp.join(self.raw_dir,'zip','cat-edge-Cooking.zip')]

    def unzip(self):
        import zipfile
        for __zip__ in self.zip_dir:
            with zipfile.ZipFile(__zip__, 'r') as zip_ref:
                zip_ref.extractall(self.raw_dir)
    
    # def continue_check(self):
        
        
    def process(self):
        print(f'root: {self.root}')
        print('processing... begin')

        self.unzip()
        
        hyperedge_label_identities,hyperedges,hyperedge_labels,node_labels = self.__load__()
        if len(hyperedge_labels) != len(hyperedges):
            print('error:  len(hyperedge_labels) != len(hyperedges)')
            exit()
        
        if max(hyperedge_labels) != len(hyperedge_label_identities):
            print('error:  max(hyperedge_labels) != len(hyperedge_label_identities)')
            print(f'len(hyperedge_label_identities)  = {len(hyperedge_label_identities)}')
            print(f'max(hyperedge_labels) = {max(hyperedge_labels)}')
            exit()
        

        need_remove_idex = []
        for idx,e in enumerate(hyperedges):
            if len(e) < self.hyperedge_size_lowbound:
                need_remove_idex.append(idx)
        for idx in need_remove_idex[::-1]:
            hyperedges.pop(idx)
            hyperedge_labels = np.delete(hyperedge_labels,idx)  



        ### remap node id.
        node_ids = dict()
        for i, edge in enumerate(hyperedges):
            _temp_ = set()
            for n_id in edge:
                if n_id not in node_ids:
                    node_ids[n_id] = len(node_ids)
                _temp_.add(node_ids[n_id])
            hyperedges[i] = _temp_

        idx_map = [ -1 for i in range(len(node_ids))]

        for k in node_ids.keys():
            idx_map[node_ids[k]] = k

        node_labels = [node_labels[i] for i in idx_map]
            # labels = labels[idx_map]
            # features = features[idx_map]


        ################


        max_node_id = max([max(e) for e in hyperedges])
        min_node_id = min([min(e) for e in hyperedges])
        if min_node_id != 0:
            print('error:  min_node_id != 0')
            print(f' min_node_id =  {min_node_id}')
            exit()
            
        if len(node_labels) != max_node_id + 1:
            print('error:  len(node_labels) != max_node_id + 1')
            print(f'len(node_labels)  = {len(node_labels)}')
            print(f'max_node_id = {max_node_id}')
            exit()

        data = HeteroData()
        hyperedge_ids = []
        node_ids = []
        for idx, nodes in enumerate(hyperedges):
            hyperedge_ids += [idx] * len(nodes)
            node_ids += nodes
            
        data['node', 'in','hyperedge'].edge_index = torch.LongTensor(np.array([node_ids, hyperedge_ids], dtype = np.int64))
        # data['hyperedge', 'node'].edge_index = torch.LongTensor(np.array([hyperedge_ids, node_ids], dtype = np.int64))
        data['hyperedge'].num_nodes = len(hyperedges)
        data['node', 'in','hyperedge'].num_edge_index = len(node_ids)
        # data['hyperedge'].labels = new_simplex_labels  
        data['hyperedge'].hyperedge_labels = torch.LongTensor(hyperedge_labels)
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
            
            
        



