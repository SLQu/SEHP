import torch

import sys,os
sys.path.append(str(os.getcwd()).split('Hyperedge')[0]+'Hyperedge')
from hedge import CACHE_ROOT,LOGS_ROOT
import hedge

@torch.no_grad()
def test(args,encoder,aggregator,classify,loader,data,evaluator,device):
    negative_keys = args.negative_keys
    encoder.eval()
    classify.eval()
    


    neg_result = {}
    for n_key in negative_keys.split('_'):
        neg_result[f"{n_key}"] = []
    all_positive_pre = []
    
    for batch_idx, batch_data in enumerate(loader):
        batch_data = batch_data.to(device)
        x_dict = encoder(batch_data)
        
        positive_hyperedge = batch_data['node', 'in','hyperedge'].edge_index
        positive_hyperedge_emb = aggregator(x_dict['node'][positive_hyperedge[0]],positive_hyperedge[1])
        _positive_pre_ = classify(positive_hyperedge_emb)
        all_positive_pre.append(_positive_pre_)

        if args.dataset_name in ['Recipe100k', 'Recipe200k', 'TencentBiGraph', 'AmazonBook','WalmartTrips']:
            _edge_index_neg_ = hedge.utils.tools.negative_generate_by_positive(batch_data['node'].num_nodes, args.negative_sample_amount, positive_hyperedge)
            negativehyperedge_emb = aggregator(x_dict['node'][_edge_index_neg_[0]],_edge_index_neg_[1])
            negative_pre = classify(negativehyperedge_emb)
            neg_result["sns"].append(negative_pre)
            # if batch_idx == 0:
                # print('  -------- test  ---------------')
                # print(f"{_positive_pre_[:5]}")
                # print( positive_hyperedge_emb[:5,:5])
                # print(f"{negative_pre[:5]}")
                # print( negativehyperedge_emb[:5,:5])

        

    if args.dataset_name not in ['Recipe100k', 'Recipe200k', 'TencentBiGraph', 'AmazonBook','WalmartTrips']:
        data = data.to(device)
        x_dict = encoder(data)
        for n_key in negative_keys.split('_'):
            
            negative_hyperedge = data[f'{n_key}_hyperedge']
            negativehyperedge_emb = aggregator(x_dict['node'][negative_hyperedge[0]],negative_hyperedge[1])
            negative_pre = classify(negativehyperedge_emb)
            neg_result[f"{n_key}"].append(negative_pre)


    #################################################################        
    positive_pre = torch.cat(all_positive_pre,0)
    all_positive_label = torch.ones_like(positive_pre)
    #############


    test_result = {}
    
    for n_key in negative_keys.split('_'):
        neg_result[f"{n_key}"] = torch.cat(neg_result[f"{n_key}"],0)
        neg_result[f"{n_key}_label"] =torch.zeros_like(neg_result[f"{n_key}"]) 

        all_pre = torch.cat((positive_pre,neg_result[f"{n_key}"]),0)
        all_label = torch.cat((all_positive_label,neg_result[f"{n_key}_label"]),0)

        result = evaluator(all_pre, all_label)
        result['pos'] = sum(positive_pre)/len(positive_pre)
        result['neg'] = sum(negative_pre)/len(negative_pre)
        result['diff'] = result['pos'] - result['neg']
        test_result[n_key] = result

    #################################################################

    
    return test_result



@torch.no_grad()
def get_epoch_score(args,encoder,aggregator,classify,loader,data,evaluator,device):
    negative_keys = args.negative_keys
    encoder.eval()
    classify.eval()
    
    all_positive_pre = []
    all_negative_pre = []
    
    for batch_idx, batch_data in enumerate(loader):
        batch_data = batch_data.to(device)
    
        x_dict = encoder(batch_data)
        
        
        positive_hyperedge = batch_data['node', 'in','hyperedge'].edge_index
        positive_hyperedge_emb = aggregator(x_dict['node'][positive_hyperedge[0]],positive_hyperedge[1])
        positive_pre = classify(positive_hyperedge_emb)
        all_positive_pre.append(positive_pre)
        

        if args.dataset_name in ['Recipe100k', 'Recipe200k', 'TencentBiGraph', 'AmazonBook','WalmartTrips']:
            _edge_index_neg_ = hedge.utils.tools.negative_generate_by_positive(batch_data['node'].num_nodes, args.negative_sample_amount, positive_hyperedge)
            negativehyperedge_emb = aggregator(x_dict['node'][_edge_index_neg_[0]],_edge_index_neg_[1])
            negative_pre = classify(negativehyperedge_emb)
            all_negative_pre.append(negative_pre)

        

    if args.dataset_name not in ['Recipe100k', 'Recipe200k', 'TencentBiGraph', 'AmazonBook','WalmartTrips']:
        data = data.to(device)
        x_dict = encoder(data)
        for n_key in negative_keys.split('_'):
            
            negative_hyperedge = data[f'{n_key}_hyperedge']
            negativehyperedge_emb = aggregator(x_dict['node'][negative_hyperedge[0]],negative_hyperedge[1])
            negative_pre = classify(negativehyperedge_emb)
            all_negative_pre.append(negative_pre)



        # data = data.to(device)
        # x_dict = encoder(data)
        
        # ns_size = len(negative_keys)
        
        # for n_key in negative_keys.split('_'):
            
        #     negative_hyperedge = data[f'{n_key}_hyperedge']
        #     # length = int(negative_hyperedge[0].shape[0]/ns_size)
        #     # negative_hyperedge =  [ids[:length] for ids in negative_hyperedge]
            
        #     negativehyperedge_emb = aggregator(x_dict['node'][negative_hyperedge[0]],negative_hyperedge[1])
        #     negative_pre = classify(negativehyperedge_emb)
    
        #     all_negative_pre.append(negative_pre)
        
    all_positive_pre = torch.cat(all_positive_pre,0)
    all_negative_pre = torch.cat(all_negative_pre,0)
    
    return all_positive_pre, all_negative_pre


