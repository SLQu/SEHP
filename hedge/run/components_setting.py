from .set_rest  import get_aggregation,get_classify,get_optimizer,get_criterion

from  .set_encoder_sampling import    get_negative_sampler,get_encoder

def components_setting(args):
    in_channels = args.in_channels
    out_channels = args.out_channels
    hidden_channels = args.hidden_channels
    classify_layer = args.classify_layer
    num_layers = args.num_layers
    aggr_type = args.aggr_type
    device = args.device
    negative_sample_mode = args.negative_sample_mode
    negative_sample_amount = args.negative_sample_amount
    lr =  args.lr
    dropout =  args.dropout
    num_node = args.num_node
    
    
    encoder = get_encoder(args,in_channels,
              out_channels,
              hidden_channels,
              num_layers = num_layers,
              device = device,
              dropout = dropout
              )
    
    aggregator = get_aggregation(args,aggr_type,device)
    classify = get_classify(args,classify_layer,device)
    
    
    
    D_optimizer = get_optimizer(args,[encoder, classify],lr)
    
    
    if args.negative_sample_mode == 'gns' :
        # if args.share_encoder:
        #     sampler_encoder = encoder
        # else:
       
        sampler_encoder = get_encoder(args,in_channels,
                out_channels,
                hidden_channels,
                num_layers = num_layers,
                device = device,
                dropout = dropout
                )
        
        sampler_decoder = get_encoder(args,out_channels,
                1,
                hidden_channels,
                num_layers = num_layers,
                device = device,
                dropout = dropout,
                model_name = 'mlp'
                )
        
        negative_sampler = get_negative_sampler(args,negative_sample_mode,negative_sample_amount,device,encoder = sampler_encoder, decoder = sampler_decoder)
         
        G_optimizer = get_optimizer(args,[sampler_encoder, sampler_decoder],lr)

    elif args.negative_sample_mode == 'dgns' :
       
        # if args.share_encoder:
        #     sampler_encoder = encoder
        # else:
        sampler_encoder = get_encoder(args,in_channels,
                    out_channels,
                    hidden_channels,
                    num_layers = num_layers,
                    device = device,
                    dropout = dropout
                    )
        
        sampler_decoder = get_encoder(args,out_channels * 2,
                1,
                hidden_channels,
                num_layers = num_layers,
                device = device,
                dropout = dropout,
                model_name = 'mlp'
                )
        
        negative_sampler = get_negative_sampler(args,negative_sample_mode,negative_sample_amount,device, aggregator = aggregator, encoder = sampler_encoder, decoder = sampler_decoder)
         
        G_optimizer = get_optimizer(args,[sampler_encoder, sampler_decoder],lr)

    elif args.negative_sample_mode == 'dgns2' :
       
        sampler_encoder = get_encoder(args,in_channels,
                    out_channels,
                    hidden_channels,
                    num_layers = num_layers,
                    device = device,
                    dropout = dropout
                    )
        
        sampler_decoder = get_encoder(args,out_channels * 2,
                out_channels,
                hidden_channels,
                num_layers = num_layers,
                device = device,
                dropout = dropout,
                model_name = 'mlp'
                )
        
        sampler_decoder2 = get_encoder(args,out_channels * 2,
                1,
                hidden_channels,
                num_layers = num_layers,
                device = device,
                dropout = dropout,
                model_name = 'mlp'
                )
        
        negative_sampler = get_negative_sampler(args,negative_sample_mode,negative_sample_amount,device, aggregator = aggregator, encoder = sampler_encoder, decoder = [sampler_decoder,sampler_decoder2])
         
        G_optimizer = get_optimizer(args,[sampler_encoder, sampler_decoder,sampler_decoder2],lr)

    
        
    elif args.negative_sample_mode == 'dgns_epre' :
       

        sampler_encoder = get_encoder(args,in_channels,
                    out_channels,
                    hidden_channels,
                    num_layers = num_layers,
                    device = device,
                    dropout = dropout
                    )
        
        sampler_decoder = get_encoder(args,out_channels * 2,
                out_channels,
                hidden_channels,
                num_layers = 2,
                device = device,
                dropout = dropout,
                model_name = 'mlp'
                )

        alignment_layer = get_encoder(args,out_channels,
                out_channels,
                hidden_channels,
                num_layers = 1,
                device = device,
                dropout = dropout,
                model_name = 'mlp',   # res_net
                )
        
        negative_sampler = get_negative_sampler(args,negative_sample_mode,negative_sample_amount,device, aggregator = aggregator, encoder = sampler_encoder, decoder = [sampler_decoder,alignment_layer])
         
        G_optimizer = get_optimizer(args,[sampler_encoder, sampler_decoder,alignment_layer],lr)
    elif args.negative_sample_mode == 'fgns':
       
        sampler_encoder = None
        
        dim = [64, 128,256, args.negative_num_nodes]
        sampler_decoder = get_encoder(args,128,
                num_node,
                hidden_channels,
                num_layers = dim,
                device = device,
                dropout = dropout,
                model_name = 'mlp_generator'
                )
        
        negative_sampler = get_negative_sampler(args,negative_sample_mode,negative_sample_amount,device, encoder = sampler_encoder, decoder = sampler_decoder)
         
        G_optimizer = get_optimizer(args,[sampler_decoder],lr)
        
    elif args.negative_sample_mode == 'pgns':
       
        sampler_encoder = get_encoder(args,in_channels,
            1,
            hidden_channels,
            num_layers = num_layers,
            device = device,
            dropout = dropout
            )
        
        dim = [args.negative_num_nodes, 256,256, args.negative_num_nodes]
        sampler_decoder = get_encoder(args,args.negative_num_nodes,
                num_node,
                hidden_channels,
                num_layers = dim,
                device = device,
                dropout = dropout,
                model_name = 'mlp_generator'
                )
        
        negative_sampler = get_negative_sampler(args,negative_sample_mode,negative_sample_amount,device, encoder = sampler_encoder, decoder = sampler_decoder)
         
        G_optimizer = get_optimizer(args,[sampler_decoder],lr)
        
    else:
        negative_sampler = get_negative_sampler(args,negative_sample_mode,negative_sample_amount,device)
        sampler_encoder,G_optimizer = None,None
    
    criterion = get_criterion(args)
    
    return encoder,aggregator,classify,negative_sampler,D_optimizer,criterion,G_optimizer