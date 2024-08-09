from hedge._global import CACHE_ROOT
from .cornell_base import CornellBase
from hedge.utils import format_and_sort_list
from hedge.utils import convert_bipartite_to_list_hyperedges
import os
import os.path as osp
import pickle
import torch
import numpy as np
from torch_geometric.data import Data,HeteroData
from torch_sparse import coalesce
import random

from .abstract_dataset import AbstractDataset


from dhg import Graph, Hypergraph
from dhg.data import CoauthorshipCora
from dhg.data import CoauthorshipDBLP 
from dhg.data import CocitationCora
from dhg.data import CocitationCiteseer
from dhg.data import CocitationPubmed
from dhg.data import YelpRestaurant
from dhg.data import WalmartTrips
from dhg.data import HouseCommittees
from dhg.data import News20
from dhg.data import DBLP4k
from dhg.data import DBLP8k
from dhg.data import IMDB4k
from dhg.data import Recipe100k
from dhg.data import Recipe200k
from dhg.data import Yelp3k
from dhg.data import Tencent2k
from dhg.data import Cooking200
from dhg.data import TencentBiGraph
from dhg.data import AmazonBook





'''

CoauthorshipCora  ['num_classes', 'num_vertices', 'num_edges', 'dim_features', 'features', 'edge_list', 'labels', 'train_mask', 'val_mask', 'test_mask'])
CoauthorshipDBLP  ['num_classes', 'num_vertices', 'num_edges', 'dim_features', 'features', 'edge_list', 'labels', 'train_mask', 'val_mask', 'test_mask'])
CocitationCora    ['num_classes', 'num_vertices', 'num_edges', 'dim_features', 'features', 'edge_list', 'labels', 'train_mask', 'val_mask', 'test_mask'])
CocitationCiteseer['num_classes', 'num_vertices', 'num_edges', 'dim_features', 'features', 'edge_list', 'labels', 'train_mask', 'val_mask', 'test_mask'])
CocitationPubmed  ['num_classes', 'num_vertices', 'num_edges', 'dim_features', 'features', 'edge_list', 'labels', 'train_mask', 'val_mask', 'test_mask'])
YelpRestaurant    ['num_classes', 'num_vertices', 'num_edges', 'dim_features', 'features', 'edge_list', 'labels', 'state', 'city'])
News20            ['num_classes', 'num_vertices', 'num_edges', 'dim_features', 'features', 'edge_list', 'labels'])
Recipe100k        ['num_classes', 'num_vertices', 'num_edges', 'dim_features', 'features', 'edge_list', 'labels'])
Recipe200k        ['num_classes', 'num_vertices', 'num_edges', 'dim_features', 'features', 'edge_list', 'labels'])
Yelp3k            ['num_classes', 'num_vertices', 'num_edges', 'dim_features', 'features', 'edge_list', 'labels'])
Tencent2k         ['num_classes', 'num_vertices', 'num_edges', 'dim_features', 'features', 'edge_list', 'labels', 'train_mask', 'val_mask', 'test_mask'])


Cooking200        ['num_classes', 'num_vertices', 'num_edges',                             'edge_list', 'labels', 'train_mask', 'val_mask', 'test_mask'])
DBLP8k            [               'num_vertices', 'num_edges',                             'edge_list'])
WalmartTrips      ['num_classes', 'num_vertices', 'num_edges',                             'edge_list', 'labels'])
HouseCommittees   ['num_classes', 'num_vertices', 'num_edges',                             'edge_list', 'labels'])


IMDB4k            ['num_classes', 'num_vertices', 'num_director_edges', 'num_actor_edges', 'dim_features', 'features', 'labels', 'edge_by_director', 'edge_by_actor'])
DBLP4k            ['num_classes', 'num_vertices', 'num_paper_edges', 'num_term_edges', 'num_conf_edges', 'dim_features', 'features', 'labels', 'edge_by_paper', 'edge_by_term', 'edge_by_conf', 'paper_author_dict', 'term_paper_dict', 'conf_paper_dict'])
        


        


CoauthorshipCora, CoauthorshipDBLP, CocitationCora, CocitationCiteseer, CocitationPubmed, YelpRestaurant, News20, Recipe100k, Recipe200k, Yelp3k, Tencent2k, 
Cooking200, DBLP8k, WalmartTrips, HouseCommittees,
IMDB4k, DBLP4k

------------------------------------
                                    'num_vertices', 'num_edges', 'edge_list',   'dim_features', 'features'
Tencent2k,           tencent_2k          ,   2146,    6378,   9.7904
CoauthorshipCora,    coauthorship_cora   ,   2708,    1072,   4.2771
CocitationCora,      cocitation_cora     ,   2708,    1579,   3.0310
CocitationCiteseer,  cocitation_citeseer ,   3312,    1079,   3.2002
Yelp3k,              yelp_3k             ,   3855,   24137,   5.4400
CocitationPubmed,    cocitation_pubmed   ,  19717,    7963,   4.3487
CoauthorshipDBLP,    coauthorship_dblp   ,  41302,   22363,   4.4520
News20,              20news              ,  16342,     100,   654.5100
YelpRestaurant,      yelp_restaurant     ,  50758,  679302,   6.6592
Recipe100k,          recipe-100k-v2      , 101585,   12387,   206.8980
Recipe200k,          recipe-200k-v2      , 240094,   18129,   243.4836


Tencent2k, CoauthorshipCora, CocitationCora, CocitationCiteseer, Yelp3k, CocitationPubmed, CoauthorshipDBLP, News20, YelpRestaurant, Recipe100k, Recipe200k


                                'num_vertices', 'num_edges', 'edge_list'
HouseCommittees,     house_committees    ,   1290,     341,   34.7889
Cooking200           cooking_200         ,   7403,    2755,   19.9564
DBLP8k,              dblp_8k             ,   8657,    2603,   4.5125
WalmartTrips,        walmart_trips       ,  88860,   69906,   6.5893



'''


class DGH_dataset(AbstractDataset):
    
    
    def __init__(self, datasetname = '_dgn_',root = '',
                 hyperedge_size_lowbound = 2,ratio = [0.6, 0.2, 0.2]) -> None:
        
        '''
        root and dataset should be defined before super().__init__(),
        '''
        # citeseer  cora     pubmed
        self.datasetname = datasetname
        root = CACHE_ROOT if root == '' else root
        self.hyperedge_size_lowbound = hyperedge_size_lowbound
        self.hyperedge_size_upbound = 2000
        super().__init__(root,ratio)
        
        self.data_list = self.processed_file_names



    @property
    def raw_file_names(self):
        return ''
    
    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, 'dgh_processed', self.datasetname,f'hyperedge_size_lowbound_{self.hyperedge_size_lowbound}',self.ratio_str)
    
    def _get_data_(self):


        # CoauthorshipCora, CoauthorshipDBLP, CocitationCora, CocitationCiteseer, CocitationPubmed, YelpRestaurant, News20, Recipe100k, Recipe200k, Yelp3k, Tencent2k, 

        if self.datasetname == 'CoauthorshipCora':
            return CoauthorshipCora()
        elif self.datasetname == 'CoauthorshipDBLP':
            return CoauthorshipDBLP()
        elif self.datasetname == 'CocitationCora':
            return CocitationCora()
        elif self.datasetname == 'CocitationCiteseer':
            return CocitationCiteseer()
        elif self.datasetname == 'CocitationPubmed':
            return CocitationPubmed()
        elif self.datasetname == 'YelpRestaurant':
            return YelpRestaurant()
        elif self.datasetname == 'News20':
            return News20()
        elif self.datasetname == 'Recipe100k':
            return Recipe100k()
        elif self.datasetname == 'Recipe200k':
            return Recipe200k()
        elif self.datasetname == 'Yelp3k':
            return Yelp3k()
        elif self.datasetname == 'Tencent2k':
            return Tencent2k()
        elif self.datasetname == 'TencentBiGraph':
            return TencentBiGraph()
        elif self.datasetname == 'Cooking200':
            return Cooking200()
        elif self.datasetname == 'DBLP8k':
            return DBLP8k()
        elif self.datasetname == 'WalmartTrips':
            return WalmartTrips()
        elif self.datasetname == 'HouseCommittees':
            return HouseCommittees()
        elif self.datasetname == 'AmazonBook':
            return AmazonBook()

            
            
    def process(self):
        print(f'root: {self.root}')
        print('processing... begin')

        data = self._get_data_()

        if self.datasetname in ['TencentBiGraph']:
            edge_list = torch.LongTensor(data['edge_list']).T
            raw_hedge = convert_bipartite_to_list_hyperedges(edge_list)
            raw_hedge = [ e.tolist()  for e in raw_hedge]
        elif self.datasetname in ['AmazonBook']:
            raw_hedge = data['train_adj_list'] + data['test_adj_list']

        else:
            raw_hedge = data['edge_list']
        
        # filter repeated hyperedges and hyperedges with size less than 2
        target_hedge = dict()

        removed_hedge = []
        for idx,e in enumerate(raw_hedge):
            if  len(e) < self.hyperedge_size_lowbound:
                removed_hedge.append([idx,e])
                continue
            
            if  len(e) > self.hyperedge_size_upbound:
                removed_hedge.append([idx,e])
                continue


            if type(e) is torch.Tensor:
                e = e.tolist()
            elif type(e) is set:
                e = list(e)
            elif type(e) is tuple:
                e = list(e)
            
            for i, n_id in enumerate(e):
                e[i] = int(n_id)

            _key_ = format_and_sort_list(e)
            if _key_ not in target_hedge.keys():
                target_hedge[_key_] = (idx,e)
        print(f'number of target hyperedges: {len(target_hedge)}')
        print(f'number of removed hyperedges: {len(removed_hedge)}')
        # print(removed_hedge)
 
        ### remap node id.
        _hedge_ = [ target_hedge[k][1] for k in target_hedge]
        node_ids = dict()
        for i, edge in enumerate(_hedge_):
            _temp_ = set()
            for n_id in edge:
                if n_id not in node_ids:
                    node_ids[n_id] = len(node_ids)
                _temp_.add(node_ids[n_id])
            _hedge_[i] = _temp_

        idx_map = [ -1 for i in range(len(node_ids))]

        for k in node_ids.keys():
            idx_map[node_ids[k]] = k

        # node_labels = [node_labels[i] for i in idx_map]
            

        if self.datasetname in ['Cooking200', 'DBLP8k', 'WalmartTrips', 'HouseCommittees', 'AmazonBook']:
            features = [torch.eye(len(idx_map))]
        elif self.datasetname in ['TencentBiGraph']:
            features = [data['u_features'][i] for i in idx_map]


        else:
            features = [data['features'][i] for i in idx_map]

        ################
               
        data = HeteroData()
        hyperedge_ids = []
        node_ids = []
        for idx, nodes in enumerate(_hedge_):
            hyperedge_ids += [idx] * len(nodes)
            node_ids += nodes
            
        data['node', 'in','hyperedge'].edge_index = torch.LongTensor(np.array([node_ids, hyperedge_ids], dtype = np.int64))
        # data['hyperedge', 'node'].edge_index = torch.LongTensor(np.array([hyperedge_ids, node_ids], dtype = np.int64))
        data['hyperedge'].num_nodes = len(_hedge_)
        data['node', 'in','hyperedge'].num_edge_index = len(node_ids)
        data['node'].num_nodes = len(idx_map)
        data['node'].x = torch.cat(features).view(data['node'].num_nodes, -1)
        data['node'].num_features = data['node'].x.shape[1]

        # data['node'].x = torch.nn.Embedding(num_embeddings = data['node'].num_nodes, embedding_dim = 333)
        # data['node'].x = torch.eye(data['node'].num_nodes)  

        #######################################################
        print(f'begin spliting hyperedges')
        data = self.hyperedge_split(data, mode = 'random')
        print(f'end spliting hyperedges')
        data = self.set_negative(data)
        #######################################################
        torch.save(data, osp.join(self.processed_dir, self.processed_file_names[0]))
        print('processing... end')
            
            
        



