import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class Classify(nn.Module):
    def __init__(self, layers = [256, 128, 8, 1]):
        super().__init__()
        Layers = []
        for i in range(len(layers)-1):
            Layers.append(nn.Linear(layers[i], layers[i+1]))
            if i != len(layers)-2:
                Layers.append(nn.ReLU(True))
                
        self.cls = nn.Sequential(*Layers)
        
    def reset_parameters(self):
        for layer in self.cls:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
    
    def forward(self, embedding, weights = None):
        embedding = torch.linalg.norm(embedding.unsqueeze(0),dim=0) 
        pred = self.cls(embedding)
        pred = torch.sigmoid(pred)
        return pred
    