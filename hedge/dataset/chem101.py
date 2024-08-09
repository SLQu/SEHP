
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


class Chem101(AbstractDataset):
    
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
        self.datasetname = 'chem_101'
        root = CACHE_ROOT if root == '' else root
        
        super().__init__(root,transform, pre_transform, pre_filter,ratio)
        
        self.data_list = self.processed_file_names
    
    @property
    def raw_file_names(self):
        return ['social-chem-101.v1.0.tsv']
    
    @property
    def raw_dir(self) -> str:
        return osp.join(self.root,self.datasetname)

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, 'processed',self.datasetname)
    
    
    def get_country(self):
        
        cun =  ["Andorra", "Armenia", "Algeria", "Azerbaijan", "Austria", "Antigua and Barbuda", "Argentina", "Albania", "Angola", "Afghanistan", "Australia", "Belize", "Botswana", "Bulgaria", "Belarus", "Benin", "Brazil", "Bosnia and Herzegovina", "the Bahamas", "Barbados", "Burkina Faso", "Bolivia", "Bhutan", "Bangladesh", "Bahrain", "Belgium", "Burundi", "Brunei", "Cyprus", "Cameroon", "Cabo Verde", "Cuba", "Costa Rica", "Chad", "Chile", "Canada", "Czechia", "Congo", "Comoros", "Colombia", "the Central African Republic", "China", "Cambodia", "Croatia", "the Dominican Republic", "Denmark", "Dominica", "Djibouti", "the Democratic Republic of the Congo", "Equatorial Guinea", "Ecuador", "Eritrea", "Ethiopia", "El Salvador", "Eswatini", "Estonia", "Egypt", "Fiji", "Finland", "France", "Greece", "Guinea-Bissau", "Guatemala", "Germany", "Gabon", "Georgia", "Guinea", "Guyana", "Ghana", "Gambia", "Grenada", "Hungary", "Honduras", "the Holy See", "Haiti", "India", "Ireland", "Iran", "Indonesia", "Iraq", "Italy", "Israel", "Iceland", "Japan", "Jordan", "Jamaica", "Kazakhstan", "Kiribati", "Kyrgyzstan", "Kenya", "Kuwait", "Lithuania", "Lesotho", "Latvia", "Luxembourg", "Libya", "Lebanon", "Liberia", "Laos", "Liechtenstein", "Mauritania", "Mozambique", "Mali", "Mexico", "Micronesia", "Myanmar", "the Marshall Islands", "Malawi", "Mongolia", "Moldova", "Mauritius", "Maldives", "Morocco", "Montenegro", "Malta", "Madagascar", "Malaysia", "Monaco", "North Macedonia", "New Zealand", "the Netherlands", "Niger", "Nauru", "Nepal", "Nigeria", "North Korea", "Namibia", "Norway", "Nicaragua", "Oman", "Paraguay", "Papua New Guinea", "Philippines", "Portugal", "Panama", "Palau", "Peru", "Poland", "Pakistan", "Qatar", "Rwanda", "Romania", "Russia", "Seychelles", "Syria", "Senegal", "South Sudan", "Sri Lanka", "Saudi Arabia", "Somalia", "Slovakia", "Singapore", "Samoa", "Sierra Leone", "Suriname", "Saint Vincent and the Grenadines", "Switzerland", "Sao Tome and Principe", "the State of Palestine", "Saint Kitts and Nevis", "Sweden", "the Solomon Islands", "South Korea", "Spain", "San Marino", "Sudan", "Saint Lucia", "South Africa", "Serbia", "Slovenia", "Thailand", "Trinidad and Tobago", "Togo", "Tunisia", "Tonga", "Turkmenistan", "Turkey", "Tanzania", "Tuvalu", "Timor-Leste", "Tajikistan", "Uruguay", "the United States of America", "Uzbekistan", "Uganda", "the United Kingdom", "Ukraine", "the United Arab Emirates", "Vietnam", "Venezuela", "Vanuatu", "Yemen", "Zambia", "Zimbabwe"]
        # cun =  ['Armenia','Austria', 'the united states of america', 'the united kingdom'    , 'canada'     , 'saudi arabia'    , 'iran'    , 'afghanistan'      , 'india'   , 'qatar' ,'mexico' ,'the united arab emirates', 'china'  ]


        cun = [i.lower() for i in cun]
        return cun
    
    def process(self):
          
        print(f'root: {self.root}')
        print('processing... begin')
    
    
        # get node_ids mapping
        data = pd.read_csv(osp.join(self.raw_dir, self.raw_file_names[0]), sep='\t')

        ################################################
        
        print('------------------------------------------------')
        
        '''
        action, area, rot-categorization, rot-moral-foundations, situation, rot
        
        '''
        countrys_set = []
        countries = self.get_country()
        action = list(data['action'])
        rot = list(data['rot'])
        
        
        for idx,i in enumerate(list(data['situation'])):
            for c in countries:
                if f" {c} " in i.lower():
                    print(f"----  idx:{idx},{c}, {i} ----{action[idx]}, --- {rot[idx]}")
                    countrys_set.append(c)
        
        
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



        


