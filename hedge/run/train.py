import torch 
import os

from .test import test
    
import sys,os
sys.path.append(str(os.getcwd()).split('Hyperedge')[0]+'Hyperedge')
from hedge import CACHE_ROOT,LOGS_ROOT
import hedge


import math
def training_scheduler(lam, t, T, scheduler='linear'):
    if scheduler == 'linear':
        return min(1, lam + (1 - lam) * t / T)
    elif scheduler == 'root':
        return min(1, math.sqrt(lam ** 2 + (1 - lam ** 2) * t / T))
    elif scheduler == 'geom':
        return min(1, 2 ** (math.log2(lam) - math.log2(lam) * t / T))
    

import torch.nn as nn

def compute_mean_representation(X):
    return torch.mean(X, dim=0)

class RegularizedLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.5):
        super(RegularizedLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, X_plus, X_minus):
        mu_plus = compute_mean_representation(X_plus)
        mu_minus = compute_mean_representation(X_minus)

        L_sim = (mu_plus - mu_minus).pow(2).sum()
        
        # Calculate pairwise distances between X_plus and X_minus
        dists = torch.sum((X_plus.unsqueeze(1) - X_minus.unsqueeze(0)).pow(2), dim=2)
        L_dist = -torch.mean(dists)  # Normalize the sum of distances


        return self.alpha * L_sim + self.beta * L_dist

loss_Regularized = RegularizedLoss(alpha=1.0, beta=0.5)


def write_list_to_csv(data,_epoch_,args):
    out_path = os.path.join(os.getcwd(),'example_code','score')
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    file_path = os.path.join(out_path,f"{args.dataset_name}_{args.epochs}_{args.diffusion_step}.csv")
    with open(file_path, 'a+') as f:
        f.write(f"{_epoch_},{','.join([str(i) for i in data])} \n")



def  train(args,encoder,aggregator,classify,loader,negative_sampler,optimizer,device,criterion,G_optimizer, val):
    
    epre_random_neg = False

    val_loader,evaluator,data = val
    
    epochs = args.epochs
    logger = args.logger
    model_save_path = args.model_save_path
    checkpoint = args.checkpoint
    encoder.train()
    classify.train()

    best_difference = 0
    
    for epoch  in range(epochs):
        total_loss = 0
        average_positive_pre = []
        average_negative_pre = []
        
        g_total_loss = 0
        
        if epoch % 2  > 0 and args.negative_sample_mode in ['dgns','dgns_epre','gns']: # update D 
            trainD, trainG = False, True 
        
        else: # update G
            trainD, trainG = True, False


        if trainD:
            encoder.train()
            classify.train()
            optimizer.zero_grad()
            # negative_sampler.eval()
        else:
            negative_sampler.train()
            G_optimizer.zero_grad()
            
            encoder.eval()
            classify.eval()
        
        average_positive_pre_batchs = []
        average_negative_pre_batchs = []

        hedge_nums = []

        # analysis_step = True
        analysis_step = False
        negative_hyperedge_score_for_analysis = []


        for batch_idx, batch_data in enumerate(loader):
            # print(f"batch_idx: {batch_idx}/ {epoch}/{epochs}")
            batch_data = batch_data.to(device)
            optimizer.zero_grad()
            x_dict = encoder(batch_data)

            hedge_nums.append(batch_data['hyperedge'].num_nodes)
            positive_hyperedge = batch_data['node', 'in','hyperedge'].edge_index
            positive_hyperedge_emb = aggregator(x_dict['node'][positive_hyperedge[0]],positive_hyperedge[1])
            positive_pre = classify(positive_hyperedge_emb)

            ############################
            # positive_hyperedge /= 1
            ###########################
            
            batch_data ['x_dict'] = x_dict


            if args.negative_sample_mode  in ['dgns','dgns2']:

                if analysis_step:

                    #  negative_hyperedge_emb  shape: batch_data['hyperedge'].num_nodes * args.diffusion_step, dim
                    neg_node_ids, neg_hyperedge_ids, negative_hyperedge_emb  = negative_sampler(batch_data,analysis_step=analysis_step)

                else:
                    neg_node_ids, neg_hyperedge_ids, negative_hyperedge_emb  = negative_sampler(batch_data)

                    #  negative_hyperedge_emb  shape: batch_data['hyperedge'].num_nodes * args.negative_sample_amount, dim
                
                if args.neg_hedge_rep:
                    negativehyperedge_emb = negative_hyperedge_emb
                else:
                    negativehyperedge_emb = aggregator(x_dict['node'][neg_node_ids],neg_hyperedge_ids)

            elif args.negative_sample_mode  == 'dgns_epre':
                negativehyperedge_emb  = negative_sampler(batch_data, positive_hyperedge_emb = positive_hyperedge_emb)
                # negativehyperedge_emb = torch.rand_like(negativehyperedge_emb).to(device)
                
                if epre_random_neg:
                    _edge_index_neg = hedge.utils.tools.negative_generate_by_positive(batch_data['node'].num_nodes, args.negative_sample_amount, positive_hyperedge)
                    neg_hyperedge_emb2 = aggregator(x_dict['node'][_edge_index_neg[0]],_edge_index_neg[1])
                    negative_pre2 = classify(neg_hyperedge_emb2)


            else:
                if args.dataset_name in ['Recips']:
                    negative_hyperedge = hedge.utils.tools.negative_generate_by_positive(batch_data['node'].num_nodes, args.negative_sample_amount, positive_hyperedge)
                else:
                    negative_hyperedge  = negative_sampler(batch_data)
                
                # negative_hyperedge  = negative_sampler(batch_data)


                negativehyperedge_emb = aggregator(x_dict['node'][negative_hyperedge[0]],negative_hyperedge[1])

            negative_pre = classify(negativehyperedge_emb)
            
            average_positive_pre_batchs.append(positive_pre)
            average_negative_pre_batchs.append(negative_pre)
                  
            if trainD:
                if args.ce:   # skip
                    size = training_scheduler(0.5, epoch, args.epochs,scheduler='linear')
                    positive_pre = positive_pre.squeeze()                    
                    # values,indices = torch.sort(1 - positive_pre,descending=False)
                    # posi_selected_idx = indices[:int(positive_pre.shape[0] * size)]
                    values,indices = torch.sort(negative_pre,descending=False)
                    nega_selected_idx = indices[:int(negative_pre.shape[0] * size)]
                    # loss = (-torch.mean(positive_pre[posi_selected_idx]) + torch.mean(negative_pre[nega_selected_idx]))/2

                    loss = (-torch.mean(positive_pre) + torch.mean(negative_pre[nega_selected_idx]))/2

                else:            

                    if epre_random_neg and args.negative_sample_mode  == 'dgns_epre':
                        negative_pre = torch.cat([negative_pre,negative_pre2],dim=0).squeeze()
                        # negative_pre = negative_pre2.squeeze()   #   random noise negative
                        # negative_pre = negative_pre.squeeze()
                    else:
                        negative_pre = negative_pre.squeeze()
                    
                    # only the last one is used for loss
                    if analysis_step:
                        negative_pre = negative_pre.view(-1,args.diffusion_step)[:,-1]


                    loss = (-torch.mean(positive_pre) + torch.mean(negative_pre))/2

                    
                total_loss += loss.item()
                loss.backward()
                optimizer.step()
            
            if trainG:

                if args.negative_sample_mode in ['dgns','dgns2', 'dgns_epre']:


                


                    if analysis_step:
                        negative_pre = negative_pre.view(-1,args.diffusion_step)
                        # loss_part1 = - torch.mean(negative_pre[:,:-1])   #  negative_pre is the bigger, the better
                        loss_part1 = - torch.mean(negative_pre)   #  negative_pre is the bigger, the better

                    else:
                        loss_part1 = - torch.mean(negative_pre)   #  negative_pre is the bigger, the better
                        negative_pre = negative_pre.reshape(-1,args.negative_sample_amount)


                    loss_part2 = torch.nn.functional.relu(negative_pre[:,:-1] - negative_pre[:,1:]).mean() 
                    # log(s_{i-1} / s_i)
                    # here, we use log for loss design
                    # loss_part2 = torch.mean(torch.log(negative_pre[:,:-1] / negative_pre[:,1:]))


                    loss = loss_part2 + loss_part1 * 1
                    # loss = loss_part2
                    # loss =  loss_part1


                    if args.negative_sample_mode  == 'dgns_epre':
                        loss = loss_part2 + loss_part1 + loss_Regularized(positive_hyperedge_emb, negativehyperedge_emb)
                        # loss = loss_part2 + loss_part1 
                        # loss = loss_part1
                        # loss = loss_part2

                    

                elif args.negative_sample_mode == 'gns':
                    loss = - torch.mean(negative_pre)


                g_total_loss += loss.item()
                loss.backward()
                G_optimizer.step()

            if sum(hedge_nums) > args.num_hyperedge:
                break


        average_positive_pre = torch.mean(torch.cat(average_positive_pre_batchs))
        average_negative_pre = torch.mean(torch.cat(average_negative_pre_batchs)) 

        if analysis_step:
            mean_res = torch.mean(torch.cat(average_negative_pre_batchs).view(-1, args.diffusion_step),dim=0)
            print(mean_res)
            write_list_to_csv(mean_res.tolist(),epoch,args)
            print('--------------------=================')



        
        val_all_res = test(args,encoder,aggregator,classify,val_loader,data,evaluator,device)
        
        if trainD:   
            #     accuracy       f1        auroc
            summary_score = val_all_res['sns']['precision'] + val_all_res['sns']['auroc']
            if summary_score > best_difference and checkpoint:
                best_difference = summary_score
  
                torch.save({
                    'epoch': epoch,
                    'model': encoder.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'classify': classify.state_dict(),
                    'aggregator': classify.state_dict()                
                }, os.path.join(model_save_path, f"epoch_{epoch}_best.pth"))   
                print(f"best model saved at epoch {epoch}")

        if args.dataset_name in ['Recipe100k','Recipe200k']:

            n_key = 'sns'
            n_score = val_all_res[f'{n_key}']['neg'].item()
            p_score = val_all_res[f'{n_key}']['pos'].item()
            diff = val_all_res[f'{n_key}']['diff'].item()
            auc = val_all_res[f'{n_key}']['auroc']
            ap = val_all_res[f'{n_key}']['precision']
            logger.info(f"epoch,{epoch}, total_loss,{total_loss:.4f},(pos,neg,diff)({average_positive_pre:.4f},{average_negative_pre:.4f},{average_positive_pre - average_negative_pre:.4f}), {n_key},(auc, ap):({auc:.4f}, {ap:.4f}), (pos,neg,diff): ({p_score:.4f},{n_score:.4f},{diff:.4f}) ")



        else:
            logger.info(f"epoch,{epoch},total_loss,{total_loss:.4f},pos,{average_positive_pre:.4f},neg,{average_negative_pre:.4f},diff,{average_positive_pre - average_negative_pre:.4f} ")
            for n_key in args.negative_keys.split('_'):
                
                n_score = val_all_res[f'{n_key}']['neg'].item()
                p_score = val_all_res[f'{n_key}']['pos'].item()
                diff = val_all_res[f'{n_key}']['diff'].item()
                auc = val_all_res[f'{n_key}']['auroc']
                ap = val_all_res[f'{n_key}']['precision']
                logger.info(f"epoch,{epoch}, {n_key},(auc, ap):({auc:.4f}, {ap:.4f}), (pos,neg,diff): ({p_score:.4f},{n_score:.4f},{diff:.4f}) ")



        

    if checkpoint:
        torch.save({
                    'epoch': epoch,
                    'model': encoder.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'classify': classify.state_dict(),
                    'aggregator': aggregator.state_dict()   
                }, os.path.join(model_save_path, f"epoch_last.pth")) 
           
    return None
        
