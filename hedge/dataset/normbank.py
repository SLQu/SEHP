
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

from hedge.loader import BaseLoader
import pandas as pd
from hedge.utils import get_sentence_embeddings



# l = 0: 6770
# l = 1: 5976
# l = 2: 2749


class NormBank(AbstractDataset):
    
    # 
    def __init__(self, datasetname = 'normbank',root = '',
                 transform=None, 
                 pre_transform=None, 
                 pre_filter=None,
                 ratio = [0.6, 0.2, 0.2]) -> None:
        
        '''
        root and dataset should be defined before super().__init__(),
        '''
        # citeseer  cora     pubmed
        self.datasetname = 'normbankacl'
        root = CACHE_ROOT if root == '' else root
        
        super().__init__(root,transform, pre_transform, pre_filter,ratio)
        
        self.data_list = self.processed_file_names
    
    @property
    def raw_file_names(self):
        return ['NormBank.csv']
    
    @property
    def raw_dir(self) -> str:
        return osp.join(self.root,self.datasetname)

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, 'processed',self.datasetname)
    
    def process(self):
          
        print(f'root: {self.root}')
        print('processing... begin')
    
            
        '''
        
        # print(data['setting'])
        # print(data['behavior'])
        # print(data['setting-behavior'])
        # print(data['constraints'])
        # print(data['constraints_given'])
        # print(data['constraint_predict'])
        # print(data['norm'])
        # print(data['label'])
        # print(data['split'])
         
        '''  
        # get node_ids mapping
        data = pd.read_csv(osp.join(self.raw_dir, self.raw_file_names[0]), sep=',')
        hyperedges = [ [] for i in range(len(data))]
        print(len(hyperedges))
        setting = list(data['setting'])
        print(f" original setting: {len(setting)}, unique setting: {len(set(setting))}, {len(setting)/len(set(setting)) }")
        behavior = list(data['behavior'])
        print(f" original behavior: {len(behavior)}, unique setting: {len(set(behavior))}, {len(behavior) / len(set(behavior))}")
        constraints = []
        max_len = []
        for i in list(data['constraints']):
            constraints += i.split('[AND]')
            max_len.append(len(i.split('[AND]')))
        print(f" original constraints: {len(constraints)}, unique setting: {len(set(constraints))}, {len(constraints) / len(set(constraints))}")
        print(f"max:{ max(max_len)}")
        nodes = setting + behavior + constraints
        nodes = set(nodes)
        node_text_2_ids = {node: i for i, node in enumerate(nodes)}
        print(f"node_text_2_ids: {len(node_text_2_ids)}")
        
        # get hyperedges
        splits   = []
        edge_labels   = []
        for idx, row in data.iterrows():
            setting = node_text_2_ids[row['setting']]
            behavior = node_text_2_ids[row['behavior']]
            constraints = [node_text_2_ids[constraint] for constraint in row['constraints'].split('[AND]')]
            hyperedges[idx] = [setting, behavior]+ constraints 
            splits.append(row['split'])
            edge_labels.append(row['label'])
        
        node_text = [''  for i in node_text_2_ids]
        for text, idx in node_text_2_ids.items():
            node_text[idx] = text
            
        hedge_text = [''  for i in hyperedges]
        for idx, n_ids in enumerate(hyperedges):

            _out_ = f"[SETTING] {node_text[n_ids[0]]}; "
            _out_ += f"[BEHAVIOR] {node_text[n_ids[1]]}; "
            _out_ += ', [AND] '.join([node_text[n_id] for n_id in n_ids[2:]])
                        
            hedge_text[idx] = _out_
            
        
        
        
        #####################################################
        data = HeteroData()
        
        hyperedge_ids = []
        node_ids = []
        for idx, nodes in enumerate(hyperedges):
            hyperedge_ids += [idx] * len(nodes)
            node_ids += nodes
            
        data['node', 'in','hyperedge'].edge_index = torch.LongTensor(np.array([node_ids, hyperedge_ids], dtype = np.int64))
        data['node', 'in','hyperedge'].num_edge_index = len(node_ids)
        data['hyperedge'].num_nodes = len(hyperedges)
        
        
        hyperedge_splits = []
        for s in splits:
            if s == 'train':
                hyperedge_splits.append(0)
            elif s == 'test':
                hyperedge_splits.append(1)
            elif s == 'dev':
                hyperedge_splits.append(2)
            else:
                raise ValueError(f'wrong split: {s}')
        data['hyperedge'].splits = torch.LongTensor(np.array(hyperedge_splits, dtype = np.int64))
        data['hyperedge'].text = hedge_text
        
        # 68057 59507 27859
        data['hyperedge'].label =  torch.LongTensor(np.array(edge_labels, dtype = np.int64)) 
        data['node'].num_nodes = len(node_text)
        data['node'].x = get_sentence_embeddings(node_text)
        data['node'].text = node_text
        
        
        #######################################################
        print(f'begin spliting hyperedges')
        # data = self.hyperedge_split(data, mode = 'random')
        
        data['hyperedge'].train_mask =  torch.tensor([label == 'train' for label in splits], dtype=torch.bool)
        data['hyperedge'].test_mask =  torch.tensor([label == 'test' for label in splits], dtype=torch.bool)
        data['hyperedge'].validate_mask =  torch.tensor([label == 'dev' for label in splits], dtype=torch.bool)
        
        print(f'end spliting hyperedges')
        # data = self.set_negative(data)
        #######################################################
        torch.save(data, osp.join(self.processed_dir, self.processed_file_names[0]))
        print('processing... end')


        
        
        
        


