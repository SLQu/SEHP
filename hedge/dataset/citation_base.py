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


class CitationBase(AbstractDataset):
    
    # 
    def __init__(self, datasetname = 'cora',root = '',
                 transform=None, 
                 pre_transform=None, 
                 pre_filter=None,
                 ratio = [0.6, 0.2, 0.2]) -> None:
        
        '''
        root and dataset should be defined before super().__init__(),
        '''
        # citeseer  cora     pubmed
        self.datasetname = datasetname
        root = CACHE_ROOT if root == '' else root
        
        super().__init__(root,transform, pre_transform, pre_filter,ratio)
        
        self.data_list = self.processed_file_names
    
    @property
    def raw_file_names(self):
        return ['features.pickle','labels.pickle','hypergraph.pickle']
    
    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, 'cocitation',self.datasetname)

    @property
    def processed_dir(self) -> str:
        
        
        return osp.join(self.root, 'processed','cocitation' ,self.datasetname,self.ratio_str)
    

        
        
    def __process__homogeneous(self):
        print(f'root: {self.root}')
        print('processing... begin')
        
        '''
        this will read the citation dataset from HyperGCN, and convert it edge_list to 
        [[ -V- | -E- ]
        [ -E- | -V- ]]
        '''
        print(f'Loading hypergraph dataset from hyperGCN:')
        
        # first load node features:
        with open(osp.join(self.raw_dir, self.raw_file_names[0]), 'rb') as f:
            features = pickle.load(f)
            features = features.todense()

        # second load node labels:
        with open(osp.join(self.raw_dir, self.raw_file_names[1]), 'rb') as f:
            labels = pickle.load(f)

        num_nodes, feature_dim = features.shape
        assert num_nodes == len(labels)
        print(f'number of nodes:{num_nodes}, feature dimension: {feature_dim}')

        features = torch.FloatTensor(features)
        labels = torch.LongTensor(labels)

        # The third, load hypergraph.
        with open(osp.join(self.raw_dir, self.raw_file_names[2]), 'rb') as f:
            # hypergraph in hyperGCN is in the form of a dictionary.
            # { hyperedge: [list of nodes in the he], ...}
            hypergraph = pickle.load(f)
        
        
        print(f'number of hyperedges: {len(hypergraph)}')

        edge_idx = num_nodes
        node_list = []
        edge_list = []
        for he in hypergraph.keys():
            cur_he = hypergraph[he]
            cur_size = len(cur_he)

            node_list += list(cur_he)
            edge_list += [edge_idx] * cur_size

            edge_idx += 1

        edge_index = np.array([ node_list + edge_list,
                                edge_list + node_list], dtype = np.int64)
        edge_index = torch.LongTensor(edge_index)

        data = Data(x = features,
                    edge_index = edge_index,
                    y = labels)

        # data.coalesce()
        # There might be errors if edge_index.max() != num_nodes.
        # used user function to override the default function.
        # the following will also sort the edge_index and remove duplicates. 
        total_num_node_id_he_id = edge_index.max() + 1
        data.edge_index, data.edge_attr = coalesce(data.edge_index, 
                None, 
                total_num_node_id_he_id, 
                total_num_node_id_he_id)
            
        data.num_nodes = num_nodes
        data.num_class = len(np.unique(labels.numpy()))
        data.num_hyperedges = len(hypergraph)
    
        torch.save(data, osp.join(self.processed_dir, self.processed_file_names[0]))
        print('processing... end')
        
    def __load__(self):
        with open(osp.join(self.raw_dir, self.raw_file_names[0]), 'rb') as f:
            features = pickle.load(f)
            features = features.todense()
            features = torch.FloatTensor(features)

        # second load node labels:
        with open(osp.join(self.raw_dir, self.raw_file_names[1]), 'rb') as f:
            labels = pickle.load(f)
            labels = torch.LongTensor(labels)

        # The third, load hypergraph.
        with open(osp.join(self.raw_dir, self.raw_file_names[2]), 'rb') as f:
            # hypergraph in hyperGCN is in the form of a dictionary.
            # { hyperedge: [list of nodes in the he], ...}
            hypergraph = pickle.load(f)
            
        return features, labels, hypergraph
    
    def process(self):
        
        if False:
            self.__process__homogeneous()
        
        else:        
            print(f'root: {self.root}')
            print('processing... begin')


            features, labels, hypergraph = self.__load__()

            ### remap node id.
            node_ids = dict()
            for edge_id in hypergraph.keys():
                _temp_ = set()
                for n_id in hypergraph[edge_id]:
                    if n_id not in node_ids:
                        node_ids[n_id] = len(node_ids)

                    _temp_.add(node_ids[n_id])
                
                hypergraph[edge_id] = _temp_

            idx_map = [ -1 for i in range(len(node_ids))]

            for k in node_ids.keys():
                idx_map[node_ids[k]] = k

            labels = labels[idx_map]
            features = features[idx_map]


            ################

            num_nodes, feature_dim = features.shape
            assert num_nodes == len(labels)
            print(f'number of nodes:{num_nodes}, feature dimension: {feature_dim}')
            print(f'number of hyperedges: {len(hypergraph)}')
            
            #####################################################
            data = HeteroData()
            data['node'].x = features
            data['node'].labels = labels
            hyperedge_ids = []
            node_ids = []
            for idx, nodes in enumerate(hypergraph.keys()):
                hyperedge_ids += [idx] * len(hypergraph[nodes])
                node_ids += hypergraph[nodes]
            
            data['node', 'in','hyperedge'].edge_index = torch.LongTensor(np.array([node_ids, hyperedge_ids], dtype = np.int64))
            data['node', 'in','hyperedge'].num_edge_index = len(node_ids)
            data['hyperedge'].num_nodes = len(hypergraph)
            data['node'].num_nodes = num_nodes
            data['node'].num_class = len(np.unique(labels.numpy()))
            ######################################################
            #######################################################
            print(f'begin spliting hyperedges')
            data = self.hyperedge_split(data, mode = 'random')
            print(f'end spliting hyperedges')
            data = self.set_negative(data)
            #######################################################
            torch.save(data, osp.join(self.processed_dir, self.processed_file_names[0]))
            print('processing... end')
            
            
            
        



