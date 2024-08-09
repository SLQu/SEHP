from .mlp import MLP,MLPgenerator
from .base import BaseModel
from .classify import Classify
from .nhp import NHP
from .res_net import SimpleResNet
all = {
    'MLP': MLP,
    'BaseModel': BaseModel,
    'Classify': Classify,
    'NHP': NHP,
    'MLPgenerator': MLPgenerator,
    'SimpleResNet': SimpleResNet,
}