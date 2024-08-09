from hedge._global import CACHE_ROOT
from torch_geometric.data import Dataset
import os.path as osp
import torch
import random

from .sampler import UNSSampler, SNSSampler, MNSSampler,CNSSampler, CorSampler, RWSampler, MixedSampler
class AbstractDataset(Dataset):
    r"""The AbstractDataset is a templete for all datasets.
    
    The content of the AbstractDataset dataset includes the following:

    .. note::

        The first item of each line in the ``adj_list`` is the user id, and the rest is the item id.

    Args:
        ``data_root`` (``str``, optional): The ``data_root`` has stored the data. If set to ``None``, this function will auto-download from server and save into the default direction ``~/.dhg/datasets/``. Defaults to ``None``.
    """
    def __init__(self, root = '',
                 transform=None, 
                 pre_transform=None, 
                 pre_filter=None,
                 ratio = [0.6, 0.2, 0.2]):
        
        self.ratio = ratio
        root = CACHE_ROOT if root == '' else root
        super().__init__(root = root)
        
    
    @property
    def raw_file_names(self):
        pass
    
    @property
    def ratio_str(self):
        return '_'.join([str(r) for r in self.ratio])
    
    @property
    def raw_dir(self) -> str:
        pass


    @property
    def processed_file_names(self):
        return ['data.pt']
    
    
    @property
    def processed_dir(self) -> str:
        pass
    
    def process(self):
        pass
    
    def len(self):
        return len(self.processed_file_names)
    
    def get(self, idx):
        if len(self.processed_file_names) == 1:
            data = torch.load(osp.join(self.processed_dir, self.processed_file_names[0]))
            return data
        else:
            data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
            return data 
    
    
    def set_negative(self, data):
        
        # from hedge.sample import SizedNegativeSampling
        # negative_sample_amount = 1
        # negative_model_pooling = [('snsaaa',SizedNegativeSampling)]
        
        # for n_key,sample_model in negative_model_pooling:
        #     print(f'begin {n_key} negative hyperedge sampling')
        #     negative_sampler = sample_model(amount=negative_sample_amount)
        #     negative_hyperedge  = negative_sampler(data, negative_number = int(data['hyperedge'].num_nodes * self.ratio[2]), data_perprocess = True)
        #     data[f'{n_key}_hyperedge'] = negative_hyperedge
        #     print(f'end {n_key} negative hyperedge sampling')

        incidence = dict()

        for elem in data['node','in','hyperedge'].edge_index.t().tolist() :
            v, e = elem
            e = int(e)
            v = int(v)
            if e not in incidence :
                incidence[e] = []
            incidence[e].append(v)
            
        HE = []
        for e in incidence:
            HE.append(frozenset(incidence[e]))

        pred_num = data['hyperedge'].test_mask.sum().item()

        # sample_pooling = {'uns': UNSSampler, 'sns': SNSSampler, 'mns': MNSSampler,'cns': CNSSampler, 'corrupt': CorSampler, 'rw': RWSampler, 'mixed': MixedSampler}
        sample_pooling = {'sns': SNSSampler, 'mns': MNSSampler,'cns': CNSSampler, 'mixed': MixedSampler}
        # sample_pooling = {'cns': CNSSampler, 'sns': SNSSampler}


        '''



        Recipe100k       101,585	  12,387        206.8980
        Recipe200k	     240,094 	  18,129        243.4836
        TencentBiGraph   619,030      90,044
        AmazonBook        91,599     100,000
        '''
        if self.datasetname in ['Recipe100k', 'Recipe200k', 'TencentBiGraph', 'AmazonBook', 'WalmartTrips']:
            sample_pooling = {'sns': SNSSampler}

        # sample_pooling = {'mns': MNSSampler}



        for ns in sample_pooling:
            print(ns)
            sampler_instant = sample_pooling[ns](pred_num)
            negatives = sampler_instant(set(HE))
            node_ids = []
            hyperedge_ids = []
            for idx, e in enumerate(negatives):
                negatives[idx] = list(e)
                for v in e:
                    node_ids.append(v)
                    hyperedge_ids.append(idx)

            data[f'{ns}_hyperedge'] = torch.cat( [torch.LongTensor(node_ids), torch.LongTensor(hyperedge_ids)]).view(2, -1)
        print('end negative hyperedge sampling')
        return data
    
    
    
    def hyperedge_split(self, data, mode = 'random'):
        """
        only split hyperedge for heterogeneous graph
        """

        if mode == 'random':
            train_mask, test_mask, validate_mask= __random_split__(data['hyperedge'].num_nodes, self.ratio, seed = None)
        elif mode == 'ordered':
            train_mask, test_mask, validate_mask= __ordered_split__(data['hyperedge'].num_nodes, self.ratio)
            
        data['hyperedge'].train_mask, data['hyperedge'].test_mask, data['hyperedge'].val_mask = train_mask, test_mask, validate_mask     
        return data     
    
    

import time    
        
def __random_split__(total_num, ratio = [0.6, 0.2, 0.2], seed=None):
    """
    Split a list of labels into train, test, and validate sets randomly.
    
    :param seed: Seed for randomization (optional).
    :return: Three lists of boolean values indicating whether each label belongs to each set.
    """
    
    if seed is not None:
        random.seed(seed)
    
    # print(f"current time hh:mm:ss = {time.strftime('%H:%M:%S')} --- 1 --- ")
    
    train_size = int(ratio[0] * total_num)
    test_size = int(ratio[1] * total_num)
    validate_size = total_num - train_size - test_size
    # print(f"current time hh:mm:ss = {time.strftime('%H:%M:%S')} --- 2 --- ")
    
    labels = list(range(total_num))
    # Shuffle the labels randomly
    random.shuffle(labels)
    
    # Create lists to store the split results
    # print(f"current time hh:mm:ss = {time.strftime('%H:%M:%S')} --- 3 --- ")

    train_set = labels[:train_size]
    test_set = labels[train_size:train_size + test_size]
    validate_set = labels[train_size + test_size:]
    # print(f"current time hh:mm:ss = {time.strftime('%H:%M:%S')} --- 4 --- ")
    
    # Create lists of boolean values indicating the split

    train_set = torch.LongTensor(train_set)
    train_mask = torch.zeros(total_num, dtype=torch.bool)
    train_mask[train_set] = True
    # print(f"current time hh:mm:ss = {time.strftime('%H:%M:%S')} --- 5 --- ")

    test_set = torch.LongTensor(test_set)
    test_mask = torch.zeros(total_num, dtype=torch.bool)
    test_mask[test_set] = True

    validate_set = torch.LongTensor(validate_set)
    validate_mask = torch.zeros(total_num, dtype=torch.bool)
    validate_mask[validate_set] = True
    

    # train_mask, test_mask, validate_mask = [], [], []
    # for l in list(range(total_num)):
    #     if l % 1000 == 0:
    #         print(f"{l}/{total_num}    current time hh:mm:ss = {time.strftime('%H:%M:%S')} --- 5 --- ")
    #     if l in train_set:
    #         train_mask.append(True)
    #         test_mask.append(False)
    #         validate_mask.append(False)
    #     elif l in test_set:
    #         train_mask.append(False)
    #         test_mask.append(True)
    #         validate_mask.append(False)
    #     else:
    #         train_mask.append(False)
    #         test_mask.append(False)
    #         validate_mask.append(True)
    # print(f"current time hh:mm:ss = {time.strftime('%H:%M:%S')} --- 6 --- ")
    # train_mask =  torch.tensor(train_mask, dtype=torch.bool)
    # print(f"current time hh:mm:ss = {time.strftime('%H:%M:%S')} --- 7 --- ")
    # test_mask =  torch.tensor(test_mask, dtype=torch.bool)
    # print(f"current time hh:mm:ss = {time.strftime('%H:%M:%S')} --- 8 --- ")
    # validate_mask =  torch.tensor(validate_mask, dtype=torch.bool)
    # print(f"current time hh:mm:ss = {time.strftime('%H:%M:%S')} --- 9 --- ")


    return train_mask, test_mask, validate_mask

def __ordered_split__(total_num, ratio = [0.6, 0.2, 0.2]):
    train_end = int(total_num * ratio[0])
    test_end = train_end + int(total_num * ratio[1])

    train_mask = [True] * train_end + [False] * (total_num - train_end)
    test_mask = [False] * train_end + [True] * (test_end - train_end) + [False] * (total_num - test_end)
    validate_mask = [False] * test_end + [True] * (total_num - test_end)

    return train_mask, test_mask, validate_mask

