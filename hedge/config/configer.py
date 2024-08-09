import datetime
import os

from hedge.log import set_logger


class Configer(object):
    def __init__(self, **kwargs):
        
        current_datetime = datetime.datetime.now()
        create_time = current_datetime.strftime("%Y_%m_%d__%H_%M_%S__%f")

        self.__seed__(kwargs['seed'])
    
        self.__config__ = {
            "info": {
                "version": "0.1.0",
                "name": "hedge",
                "description": "A simple and fast web framework",
                "author": "squ",
                "email": ""
                },
            "create_time": create_time,
            "LOGS_ROOT": kwargs['LOGS_ROOT']
        }
    
    def __seed__(self, seed):
        import numpy as np
        import torch
        import random
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        
        
    def get_config(self):
        return self.__config__
    
    def set_args(self, args):
        args.create_time = self.__config__['create_time']
        args.pid = os.getpid()
        
        LOGS_ROOT = self.__config__['LOGS_ROOT']
        args.log_path = f"{LOGS_ROOT}/{args.dataset}"
        
        if not os.path.exists(args.log_path):
            os.makedirs(args.log_path)
        
        args.log_name = f"{args.create_time}.log"
        log_path_name = f'{args.log_path}/{args.log_name}'
        args.logger = set_logger(log_path_name)
        
        
        
        
        self.update_args = args
        return args
    
    def update_args(self, args):
        self.__config__['args'] = args
