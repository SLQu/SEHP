
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


class NormBank_for_country(AbstractDataset):
    
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
        self.datasetname = 'normbankacl_for_country'
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
        ################################################
        
        print('------------------------------------------------')
        countrys_set = []
        for idx,i in enumerate(list(data['constraints'])):
            if 'country is' in i:
                print(f"----  idx:{idx}  ---- {i}; [behavior] {list(data['behavior'])[idx]}")
                consss = ''
                for con in i.split('[AND]'):
                    if 'country is' in con:
                        
                        consss += con.split('country is')[1].strip()
                countrys_set.append(consss)

        from collections import Counter
        counts = Counter(countrys_set)
        total = sum(counts.values())
        percentage_distribution = {k: v / total * 100 for k, v in counts.items()}
        sorted_percentage = dict(sorted(percentage_distribution.items(), key=lambda item: item[1], reverse=True))
        timess = 0
        print(f"total {len(counts)} country combination, total {total} times")
        for key, value in sorted_percentage.items():
            # print(f"{timess}, country {key}: {value:.2f}%,   {counts[key]} times")
            print(f"{timess:2d}, country: {key:<{33}}, {value:6.2f}%,   {counts[key]:3d} times")
            timess += 1
            if timess > 20:
                break
            

            
        
        
        

        print('processing... end')

        exit()



        


