
import sys,os
sys.path.append(str(os.getcwd()).split('SEHP')[0]+'SEHP')

import hedge
from hedge import CACHE_ROOT,LOGS_ROOT

from hedge.utils import get_gpu_information, Evaluation
from hedge.loader import BaseLoader
from hedge.log import set_logger
from hedge.run import train,test,components_setting,get_dataset, get_epoch_score

import torch
import numpy as np
import argparse
import random

import datetime,re



torch.autograd.set_detect_anomaly(True)

if __name__ == "__main__":

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime("%Y_%m_%d__%H_%M_%S__%f")
    parser = argparse.ArgumentParser()
    
    ##### training hyperparameter #####
    ##       Recipe100k     Recipe200k      CoauthorshipCora
    parser.add_argument("--dataset_name", type=str, default='CoauthorshipCora', help='dataset name:')
    parser.add_argument("--gpu", type=int, default=0, help='gpu number. -1 if cpu else gpu number')
    parser.add_argument("--aggr_type", type=str, default='maxmin', help='aggr_type maxmin sum  mean  min  max median')
    parser.add_argument("--batch_size", type=int, default=10, help='batch size for training') 
    parser.add_argument("--stru_diff", type=int, default=1, help='using structure information during negative sampling generation') 
    parser.add_argument("--stru_decoder", type=int, default=1, help='using structure information during negative sampling generation') 
    parser.add_argument("--sample_num_hops", type=int, default=10, help='sample_num_hops') 
    parser.add_argument("--max_sample_node", type=int, default=1000, help='max_sample_node')

  
    parser.add_argument("--ce", type=int, default=0, help='  Curriculum Learning loss. 0 for false, 1 for ture   ') 
    parser.add_argument("--neg_hedge_rep", type=int, default=0, help='  using representation of negative hedge. 0 for false, 1 for ture') 

    parser.add_argument("--negative_sample_mode", type=str, default='dgns2', help='dgns2 gns  dgns_epre, d: diffusion, g: generated, epre: edge representation ') 
    parser.add_argument("--diffusion_step", type=int, default=7, help=' the step between two negative generation') 
    parser.add_argument("--epochs", type=int, default=4, help='training epochs') 
    
    parser.add_argument("--negative_sample_amount", type=int, default=1, help=' negative sample amount') 
    parser.add_argument("--num_layers", type=int, default=4, help=' num_layers for encoder') 
    parser.add_argument("--share_encoder", type=int, default=1, help='  negative will share the same encoder as the positive')  

    parser.add_argument("--out_channels", type=int, default=128, help='out_channels for encoder') 
    parser.add_argument("--hidden_channels", type=int, default=64, help='hidden_channels for encoder') 
    parser.add_argument("--classify_layer", type=list, default=[32,1], help='classify_layer shape for classifyer') 
    parser.add_argument("--lr", type=float, default=0.001, help='learning rate') 
    parser.add_argument("--dropout", type=float, default=0.5, help='dropout  rate') 
    parser.add_argument("--test_batch_size", type=int, default=10000, help='test_batch_size') 
    parser.add_argument("--create_time", type=str, default=formatted_datetime, help='create_time') 
    parser.add_argument("--negative_keys","-nks", type=str, default='sns_mns_cns_mixed', help='negative_keys  sns_mns_cns mixed') 
    parser.add_argument("--load","-ld", type=str, default='none', help='loading name of checking point of model') 
    parser.add_argument("--load_type","-ldt", type=str, default='best', help='loading the best or last of checking point of model') 
    parser.add_argument("--training", type=int, default=1, help=' whether training or not, 1 is training, 0 is not training') 
    parser.add_argument("--checkpoint", type=int, default=1, help=' whether write checkpoint or not, 1 is write, 0 is not write') 
    parser.add_argument("--conv", type=str, default='SAGE', help=' for conv n_v and v_n, SAGE, GAT, GIN ,GINE, ResGatedGraph,GATv2,Transformer    ') 
    parser.add_argument("--heads", type=int, default=2, help=' head for GAT,GATv2,Transformer    ') 
    args = parser.parse_args()

    if args.dataset_name in ['Recipe100k','Recipe200k']:
        args.negative_keys = 'sns'


    args.command_line = ' '.join(sys.argv)
    args.pid = os.getpid()
    args.log_path = f"{LOGS_ROOT}/{args.dataset_name}"
    
    args.n_class = 2
    args.ce = False if args.ce == 0 else True
    args.neg_hedge_rep = False if args.neg_hedge_rep == 0 else True
    args.share_encoder = False if args.share_encoder == 0 else True

    
    
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)
    
    args.log_name = f"{args.create_time}.log"
    log_path_name = f'{args.log_path}/{args.log_name}'
    args.logger = set_logger(log_path_name)

    dataset = get_dataset(args)
        
    args.logger.info('done for loading dataset...')

    data = dataset[0]
    args.num_node = data['node'].num_nodes
    args.num_hyperedge = data['hyperedge'].num_nodes
    args.edge_idex_num = data['node', 'in','hyperedge'].num_edge_index
    device = 'cuda:{}'.format(args.gpu) if args.gpu != -1 else 'cpu'
    args.device = device
    args.logger.info(args)
    args.in_channels= data['node'].x.shape[1]
    args.classify_layer = [args.out_channels] + args.classify_layer
    
    args.is_sorted = True if args.negative_sample_mode in ['fgns','pgns'] else False
    
    
    if args.training:

        train_loader = BaseLoader(
            data,
            num_samples={key: [20000000] for key in data.node_types},
            # Use a batch size of 128 for sampling training nodes of type paper
            batch_size=1000,     #  args.batch_size
            input_nodes=('hyperedge', data['hyperedge'].train_mask),
            is_sorted=args.is_sorted,
            sampler = 'HyperEdgeSampler',  #  HyperEdgeSamplerV2  HyperEdgeSampler
            num_hops = args.sample_num_hops,
            max_node = args.max_sample_node
        )
        
    encoder,aggregator,classify,negative_sampler,optimizer,criterion,G_optimizer  = components_setting(args)

    if args.load == 'none':
        args.model_save_path = os.path.join(args.log_path,f'{args.create_time}_save')
        os.makedirs(args.model_save_path)
    
    if args.load != 'none':
        args.load = os.path.join(args.log_path,args.load)
        
        for model_name in os.listdir(args.load):
            
            if model_name.endswith('.pth') and args.load_type in model_name:
                args.load = os.path.join(args.load,model_name)
                args.logger.info(f'loading model from {args.load}')
                checkpoint = torch.load(args.load)
                encoder.load_state_dict(checkpoint['model'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                classify.load_state_dict(checkpoint['classify'])
                # aggregator.load_state_dict(checkpoint['aggregator'])
        args.logger.info(f'loading model from {args.load} done')

    evaluator = Evaluation()
    
    val_loader = BaseLoader(
        data,
        num_samples={key: [10000] for key in data.node_types},
        # Use a batch size of 128 for sampling training nodes of type paper
        batch_size=args.test_batch_size,
        input_nodes=('hyperedge', data['hyperedge'].val_mask),
        sampler = 'HyperEdgeSampler',
        num_hops = 10,
        max_node = args.max_sample_node
    )
    
    test_loader = BaseLoader(
        data,
        num_samples={key: [10000] for key in data.node_types},
        # Use a batch size of 128 for sampling training nodes of type paper
        batch_size=args.test_batch_size,
        input_nodes=('hyperedge', data['hyperedge'].test_mask),
        sampler = 'HyperEdgeSampler',
        num_hops = 10,
        max_node = args.max_sample_node
    )

    if args.training:
    
        args.logger.info('done for train_loader...')
        
        args.logger.info('begin training...')
        
        train_result = train(args,encoder,aggregator,classify,train_loader,negative_sampler,optimizer,device,criterion,G_optimizer,[val_loader,evaluator,data])
        args.logger.info(train_result)
        
    # del val_loader
    # del train_loader
    # del optimizer


    args.logger.info('begin testing...')
    
    if args.checkpoint:
        candidate_idx = 0
        for f in os.listdir(args.model_save_path):
            if '.pth' in f and 'best' in f:
                match = re.search(r'\d+', f)
                first_number = match.group()  
                first_number_int = int(first_number)
                candidate_idx = first_number_int if first_number_int > candidate_idx else candidate_idx
     
        load_model_name_path = os.path.join(args.model_save_path,f"epoch_{candidate_idx}_best.pth")

        checkpoint = torch.load(load_model_name_path)
        encoder.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        classify.load_state_dict(checkpoint['classify'])
        args.logger.info(f'load_model_name_path: {load_model_name_path}')
        # aggregator.load_state_dict(checkpoint['aggregator'])
        
                
    test_result = test(args,encoder,aggregator,classify,test_loader,data,evaluator,device)
#################################################################

    load_model_name_path = os.path.join(args.model_save_path,f"epoch_last.pth")
    checkpoint = torch.load(load_model_name_path)
    encoder.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    classify.load_state_dict(checkpoint['classify'])

    positive_pre, negative_pre = get_epoch_score(args,encoder,aggregator,classify,test_loader,data,evaluator,device)

    torch.save({
            'positive': positive_pre.squeeze().cpu().numpy(),
            'negative': negative_pre.squeeze().cpu().numpy()
        }, os.path.join(args.model_save_path, f"epoch_last_pre_score.pth")) 


    args.logger.info(test_result)


    test_result['data'] = args.dataset_name
    test_result['model'] = 'DGNS'
    print(test_result)
    if args.dataset_name in ['Recipe100k','Recipe200k']:
        hedge.utils.out_all(test_result)
    else:
        hedge.utils.out(test_result)
    print(f"best epoch: {candidate_idx}")
    print(args.command_line)  # args.command_line = ' '.join(sys.argv)

