
import torch 
import os
    

from hedge.model import BaseModel,MLP,MLPgenerator,SimpleResNet
from hedge.sample import SizedNegativeSampling
 
 
from hedge.sample import MotifNegativeSampling
from hedge.sample import CliqueNegativeSampling
from hedge.sample import GeneratedNegativeSampling
from hedge.sample import FullGeneratedNegativeSampling
from hedge.sample import PlusGeneratedNegativeSampling

from hedge.sample import DiffusionGeneratedNegativeSampling
from hedge.sample import EdgeRepDiffusionGeneratedNegativeSampling
from hedge.sample import RepDiffusionGeneratedNegativeSampling


def get_negative_sampler(args,negative_sample_mode,negative_sample_amount,device, aggregator = None,encoder = None, decoder = None): 
 
    if negative_sample_mode == 'sns':
        negative_sampler = SizedNegativeSampling(amount=negative_sample_amount, device=device)
    elif negative_sample_mode == 'mns':
        negative_sampler = MotifNegativeSampling(amount=negative_sample_amount, device=device)
    elif negative_sample_mode == 'cns':
        negative_sampler = CliqueNegativeSampling(amount=negative_sample_amount, device=device)
    elif negative_sample_mode == 'gns':
        negative_sampler = GeneratedNegativeSampling(amount=negative_sample_amount, device=device, encoder = encoder, decoder = decoder, stru_decoder = args.stru_decoder, share_encoder = args.share_encoder)

    elif negative_sample_mode == 'dgns':
        negative_sampler = DiffusionGeneratedNegativeSampling(aggregator, amount=negative_sample_amount, device=device, encoder = encoder, decoder = decoder, stru_diff = args.stru_diff, stru_decoder = args.stru_decoder, share_encoder = args.share_encoder, diffusion_step = args.diffusion_step)

    elif negative_sample_mode == 'dgns_epre':
        negative_sampler = EdgeRepDiffusionGeneratedNegativeSampling(aggregator, amount=negative_sample_amount, device=device, encoder = encoder, decoder = decoder, stru_diff = args.stru_diff, stru_decoder = args.stru_decoder, share_encoder = args.share_encoder, diffusion_step = args.diffusion_step)

    elif negative_sample_mode == 'dgns2':
        negative_sampler = RepDiffusionGeneratedNegativeSampling(aggregator, amount=negative_sample_amount, device=device, encoder = encoder, decoder = decoder,  stru_diff = args.stru_diff, stru_decoder = args.stru_decoder, share_encoder = args.share_encoder, diffusion_step = args.diffusion_step)



    elif negative_sample_mode == 'fgns':
        negative_sampler = FullGeneratedNegativeSampling(amount=negative_sample_amount, device=device, decoder = decoder)

    elif negative_sample_mode == 'pgns':
        negative_sampler = PlusGeneratedNegativeSampling(amount=negative_sample_amount, device=device, encoder = encoder, decoder = decoder)
    
    return negative_sampler

    
def get_encoder(args,in_channels,
              out_channels,
              hidden_channels,
              num_layers = 2,
              device = torch.device('cpu'),
              dropout = 1.0,
              model_name = 'base',
              activation = 'relu'
              ):
    
    if model_name == 'base':
        
        if args.conv in set(['bert']):
            from transformers import BertModel, BertConfig
            
            # model_name = 'bert-large-uncased'
            model_name = 'bert-base-uncased'
            nlp_model = BertModel.from_pretrained(model_name)

            # config = BertConfig.from_pretrained(model_name)
            # nlp_model = BertModel(config)
            
            
        else:
            nlp_model = None
            
            
        encoder = BaseModel(in_channels,
                        out_channels,
                        hidden_channels, 
                        num_layers = num_layers, 
                        dropout = dropout, 
                        conv = args.conv,
                        heads = args.heads,
                        nlp_model = nlp_model
                        ).to(device)
    elif model_name == 'mlp':
        encoder = MLP(in_channels,
                        out_channels,
                        hidden_channels, 
                        num_layers = num_layers, 
                        dropout = dropout, 
                        activation = activation
                        ).to(device)    
    
    elif model_name == 'res_net':
        encoder = SimpleResNet(in_channels,
                        out_channels,
                        hidden_channels, 
                        num_layers = num_layers
                        ).to(device)
    elif model_name == 'mlp_generator':
        encoder = MLPgenerator(dim = num_layers).to(device)   
        
    else:
        raise NotImplementedError
        
    return encoder
