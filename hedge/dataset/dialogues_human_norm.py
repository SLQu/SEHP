
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

from .__similar_merge__ import Merging


# l = 0: 6770
# l = 1: 5976
# l = 2: 2749


class DialoguesHumanNorm(AbstractDataset):
    
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
        self.datasetname = 'norm_dia_human'
        root = CACHE_ROOT if root == '' else root
        
        super().__init__(root,transform, pre_transform, pre_filter,ratio)
        
        self.data_list = self.processed_file_names
    
    @property
    def raw_file_names(self):
        return ['dialogues_human_social_norm.json']
    
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
        # data = pd.read_csv(osp.join(self.raw_dir, self.raw_file_names[0]), sep=',')
        
        import json
        with open(osp.join(self.raw_dir, self.raw_file_names[0]), 'r') as f:
            data = json.load(f)
        
        print(len(data))
        
        hyperedges = [ [] for i in range(len(data))]    
        norm_type_list = []
        formality_list = []
        location_list = []
        topic_list = []
        social_relation_list = []
        social_distance_list = []
        dialogue_content_list = []
        
        dialogues = []
        sns_single_combination_list = []
        sns_all_list = []
        for idx, x in enumerate(data):
            # if idx == 100:
            #     break
            
            dialogue = HumanDialogue(x)
            dialogues.append(dialogue)
            norm_type_list.append(dialogue.norm)
            formality_list.append(dialogue.formality)
            location_list.append(dialogue.location)
            topic_list.append(dialogue.topic)
            social_relation_list.append(dialogue.social_relation)
            social_distance_list.append(dialogue.social_distance)
            dialogue_content_list.append(dialogue.dialogue_sentences())
            sns_single_combination_list.append(dialogue.action)
            sns_all_list += dialogue.action
            

        print(f" original norm_type: {len(norm_type_list)}, unique norm_type: {len(set(norm_type_list))}, {len(norm_type_list)/len(set(norm_type_list)) }")
        print(f" original formality: {len(formality_list)}, unique formality: {len(set(formality_list))}, {len(formality_list)/len(set(formality_list)) }")
        print(f" original location: {len(location_list)}, unique location: {len(set(location_list))}, {len(location_list)/len(set(location_list)) }")
        print(f" original topic: {len(topic_list)}, unique topic: {len(set(topic_list))}, {len(topic_list)/len(set(topic_list)) }")
        print(f" original social_relation: {len(social_relation_list)}, unique social_relation: {len(set(social_relation_list))}, {len(social_relation_list)/len(set(social_relation_list)) }")
        print(f" original social_distance: {len(social_distance_list)}, unique social_distance: {len(set(social_distance_list))}, {len(social_distance_list)/len(set(social_distance_list)) }")
        print(f" original dialogue_content: {len(dialogue_content_list)}, unique dialogue_content: {len(set(dialogue_content_list))}, {len(dialogue_content_list)/len(set(dialogue_content_list)) }")
        print(f" original sns_all : {len(sns_all_list)}, unique sns_all: {len(set(sns_all_list))}, {len(sns_all_list)/len(set(sns_all_list)) }")

        # sns_all_list  = sns_all_list[:5000]
        merge = Merging()
        merged_sentences, sentence_mapping_oid_newid = merge.run(sns_all_list, threshold=0.95)
        
        new_sns_all_list = dict()
        for i in range(len(sns_all_list)):
            new_sns_all_list[sns_all_list[i]] = merged_sentences[sentence_mapping_oid_newid[i]]
            
        
        for i, sns in enumerate(sns_single_combination_list):
            singel_new_sns = []
            for j, s in enumerate(sns):
                singel_new_sns.append(new_sns_all_list[s])
            singel_new_sns = list(set(singel_new_sns))
            sns_single_combination_list[i] = singel_new_sns
            
            
        print(f" original sns_all : {len(sns_all_list)}, unique sns_all: {len(set(new_sns_all_list))}, {len(sns_all_list)/len(set(new_sns_all_list)) }")    
        del merged_sentences, sentence_mapping_oid_newid, new_sns_all_list
        
        row_sns = []
        for sns in sns_single_combination_list:
            row_sns += sns
        sns_all_list = list(set(row_sns))
        sns_to_id = {sns: i for i, sns in enumerate(sns_all_list)}
        
        sns_single_combination_id_list = [[sns_to_id[sns] for sns in sns_list] for sns_list in sns_single_combination_list]        

        # nodes = norm_type_list + formality_list + location_list + topic_list + social_relation_list + social_distance_list + dialogue_content_list
        nodes = norm_type_list + formality_list + location_list + topic_list + social_relation_list + social_distance_list  
        nodes = set(nodes)
        node_text_2_ids = {node: i for i, node in enumerate(nodes)}
        print(f"node_text_2_ids: {len(node_text_2_ids)}")
        
        # get hyperedges
        hedge_text = [''  for i in hyperedges]
        hedge_dia = [''  for i in hyperedges]
        for idx, row in enumerate(dialogues):

            norm_type = node_text_2_ids[row.norm]
            formality = node_text_2_ids[row.formality]
            location = node_text_2_ids[row.location]
            topic = node_text_2_ids[row.topic]
            social_relation = node_text_2_ids[row.social_relation]
            social_distance = node_text_2_ids[row.social_distance]
            hyperedges[idx] = [norm_type, formality, location, topic, social_relation, social_distance]
            hedge_text[idx] = f"[Norm Category] {row.norm}; [Formality] {row.formality}; [Location] {row.location}; [Topic] {row.topic}; [Social Relation] {row.social_relation}; [Social Distance] {row.social_distance}."
            hedge_dia[idx] = row.get_utterances()
            
            # print(f"{row.get_utterances()}")
            
            
        node_text = [''  for i in node_text_2_ids]
        for text, idx in node_text_2_ids.items():
            node_text[idx] = text
                    
        
        
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
        
        
        data['hyperedge'].sns_id = sns_single_combination_id_list
        data['hyperedge'].text = hedge_text
        data['hyperedge'].dia = hedge_dia

        data['sns'].text = sns_all_list
        data['sns'].embeddings = get_sentence_embeddings(sns_all_list)
        
        # 68057 59507 27859
        data['node'].num_nodes = len(node_text)
        data['node'].x = get_sentence_embeddings(node_text)
        data['node'].text = node_text
        
        #######################################################
        print(f'begin spliting hyperedges')
        # data = self.hyperedge_split(data, mode = 'random')
        
        data['hyperedge'].train_mask =  torch.tensor([1 == 1 for _ in range(len(hyperedges))], dtype=torch.bool)
        
        print(f'end spliting hyperedges')
        # data = self.set_negative(data)
        #######################################################
        torch.save(data, osp.join(self.processed_dir, self.processed_file_names[0]))
        print('processing... end')
        
        '''
        original norm_type: 1563, unique norm_type: 5, 312.6
        original formality: 1563, unique formality: 2, 781.5
        original location: 1563, unique location: 10, 156.3
        original topic: 1563, unique topic: 15, 104.2
        original social_relation: 1563, unique social_relation: 8, 195.375
        original social_distance: 1563, unique social_distance: 6, 260.5
        original dialogue_content: 1563, unique dialogue_content: 1515, 1.0316831683168317
        original sns_all : 11893, unique sns_all: 11384, 1.0447118763176388    
        1563    
        '''
        

 


        
class Dialogue:
    def __init__(self, data_dict):
        self.speaker_1_info = data_dict["speaker_1_info"]
        self.speaker_2_info = data_dict["speaker_2_info"]
        self.norm_rule = data_dict["norm_rule"]
        self.norm = data_dict["norm"]
        self.formality = data_dict["formality"]
        self.location = data_dict["location"]
        self.topic = data_dict["topic"]
        self.social_relation = data_dict["social_relation"]
        self.social_distance = data_dict["social_distance"]
        self.dialogue = data_dict["dialogue"]
        self.id = data_dict["id"]
        self.action = data_dict["action"][0]
        
    def to_dict(self):
        return {
            "speaker_1_info": self.speaker_1_info,
            "speaker_2_info": self.speaker_2_info,
            "norm_rule": self.norm_rule,
            "norm": self.norm,
            "formality": self.formality,
            "location": self.location,
            "topic": self.topic,
            "social_relation": self.social_relation,
            "social_distance": self.social_distance,
            "dialogue": self.dialogue,
            "id": self.id,
            "action": self.action,
            "gpt_res": self.gpt_res
        }
    
    def dialogue_sentences(self, sep=" "):
        sentence = ""
        for sentences in self.dialogue:
            sentence += sentences["utterances"] + "\n"
        return sentence
    
    def get_utterances(self):
        speakers = [self.speaker_1_info, self.speaker_2_info]
        _u_ = ""
        for item in self.dialogue:
            utterance = item['utterances']
            speaker = item['speaker']-1
            _u_ += f"{speakers[speaker]['name']}: {utterance} \n "
        
        return _u_
    
    def get_dialuge_head(self):
        _h_ = f"The norm of this conversation is {self.norm}.\n "
        _h_ += f"This conversation is between two speakers. \n "
        _h_ += f"The role and name of speaker 1 are {self.speaker_1_info['role']} and {self.speaker_1_info['name']}. \n "
        _h_ += f"The role and name of speaker 2 are {self.speaker_2_info['role']} and {self.speaker_2_info['name']}. \n "
        _h_ += f"The topic of this conversation is about {self.topic}.\n "
        _h_ += f"The formality of this conversation is {self.formality}.\n "
        _h_ += f"The location of this conversation is {self.location}.\n "
        _h_ += f"The social_relation between the two speakers is {self.social_relation}.\n "
        _h_ += f"The social_distance between the two speakers is {self.social_distance}.\n "
        return _h_
    

        
class HumanDialogue:
    def __init__(self, data_dict):
        self.norm_rule = data_dict["norm rule"]
        self.norm = data_dict["norm rule"]["norm_type"]
        self.formality = data_dict["formality"]
        self.location = data_dict["location"]
        self.topic = data_dict["topic"]
        self.social_relation = data_dict["social relation"]
        self.social_distance = data_dict["social distance"]
        self.dialogue = data_dict["dialogue"]
        self.id = data_dict["id"]
        self.action = data_dict["action"][0]
        
    
    def dialogue_sentences(self, sep=" "):
        sentence = ""
        for sentences in self.dialogue:
            sentence += sentences["utterance"] + "\n"
        return sentence
    
    def to_dict(self):
        return {
            "norm rule": self.norm_rule,
            "formality": self.formality,
            "location": self.location,
            "topic": self.topic,
            "social relation": self.social_relation,
            "social distance": self.social_distance,
            "dialogue": self.dialogue,
            "id": self.id,
            "action": self.action,
        }

    
    def get_utterances(self):
        speakers = ['说话人1', "说话人2"]
        _u_ = ""
        for idx,item in enumerate(self.dialogue):
            utterance = item['utterance']
            _u_ += f"{speakers[int(idx%2)]}: {utterance} \n "
        
        return _u_

