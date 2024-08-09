import torch 
import os
    

from hedge.model import Classify
from hedge.aggregator import MaxMinAggregation
from torch_geometric.nn.aggr.basic import MeanAggregation, MaxAggregation, MinAggregation, SumAggregation
from torch_geometric.nn.aggr import MedianAggregation
from hedge.dataset import CitationBase, NDC_substances,DAWN, NDC_substances_full,Cooking,NDC_classes
from hedge.dataset import DGH_dataset



def get_dataset(args):
    if args.dataset_name in ['citeseer','cora','pubmed']:
        dataset = CitationBase(datasetname = args.dataset_name)
    elif args.dataset_name in ['NDC_substances']:
        dataset = NDC_substances(datasetname = args.dataset_name, hyperedge_size_lowbound = 2)
    elif args.dataset_name in ['NDC_substances_full']:
        dataset = NDC_substances_full(datasetname = args.dataset_name, hyperedge_size_lowbound = 2)
    elif args.dataset_name in ['DAWN']:
        dataset = DAWN(datasetname = args.dataset_name, hyperedge_size_lowbound = 2)
    elif args.dataset_name in ['cooking']:
        dataset = Cooking(datasetname = args.dataset_name, hyperedge_size_lowbound = 2)
    elif args.dataset_name in ['NDC_classes']:
        dataset = NDC_classes(datasetname = args.dataset_name, hyperedge_size_lowbound = 2)
    
    #  CoauthorshipCora, CoauthorshipDBLP, CocitationCora, CocitationCiteseer, CocitationPubmed, YelpRestaurant, News20, Recipe100k, Recipe200k, Yelp3k, Tencent2k  ,  TencentBiGraph
    elif args.dataset_name in ['CoauthorshipCora', 'CoauthorshipDBLP', 'CocitationCora', 'CocitationCiteseer', 'CocitationPubmed', 'YelpRestaurant', 'News20', 'Recipe100k', 'Recipe200k', 'Yelp3k', 'Tencent2k', 'Cooking200', 'DBLP8k', 'WalmartTrips', 'HouseCommittees','TencentBiGraph','AmazonBook']:
        dataset = DGH_dataset(args.dataset_name) 
    else:
        args.logger.info(f'wrong dataset name... {args.dataset_name}')
        exit()  
        
    return dataset 

def get_aggregation(args = None,aggr_type = 'maxmin' ,device='cuda' if torch.cuda.is_available() else 'cpu'):
    if aggr_type == 'sum':
        aggregator = SumAggregation()  
    elif aggr_type == 'mean':
        aggregator = MeanAggregation()
    elif aggr_type == 'maxmin':
        aggregator = MaxMinAggregation()
    elif aggr_type == 'max':
        aggregator = MaxAggregation()
    elif aggr_type == 'min':
        aggregator = MinAggregation()
    elif aggr_type == 'median':
        aggregator = MedianAggregation()
    
    return aggregator.to(device)


def get_classify(args = None,classify_layer = [32,1],device = 'cuda' if torch.cuda.is_available() else 'cpu'):
    classify = Classify(classify_layer).to(device)
    return classify


def get_criterion(args):
    if args.n_class > 2:
        criterion = torch.nn.CrossEntropyLoss()
    else:
        criterion = torch.nn.BCEWithLogitsLoss()
        
    return criterion

def get_optimizer(args,models,lr, optimizer_type = 'rms'):
    
    if optimizer_type == 'rms':
        optimizer_name = torch.optim.RMSprop
    elif optimizer_type == 'adam':
        optimizer_name = torch.optim.Adam
        
    # paras = [list(m.parameters())  for m in models]
    paras = [param for m in models for param in m.parameters()]
    optimizer = optimizer_name(paras, lr=lr)
    
    return optimizer

