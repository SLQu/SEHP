
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    """ adapted from https://github.com/CUAI/CorrectAndSmooth/blob/master/gen_models.py """

    def __init__(self, in_channels,out_channels,hidden_channels =64, num_layers = 1, dropout = 1.0, activation = 'relu'):
        super().__init__()
        # in_channels = in_channels
        # hidden_channels = hidden_channels
        # out_channels = out_channels
        # num_layers = num_layers
        
        self.dropout = dropout
        self.lins = nn.ModuleList()
        
        if num_layers == 1:
                # just linear layer i.e. logistic regression
                self.lins.append(nn.Linear(in_channels, out_channels))
        else:
            self.lins.append(nn.Linear(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.lins.append(nn.Linear(hidden_channels, hidden_channels))
                # self.lins.append(self._get_activation(activation))

            self.lins.append(nn.Linear(hidden_channels, out_channels))
            # self.lins.append(self._get_activation(activation))



    # def _get_activation(self, activation):
    #     if activation == 'relu':
    #         return nn.ReLU()
    #     elif activation == 'tanh':
    #         return nn.Tanh()
    #     elif activation == 'sigmoid':
    #         return nn.Sigmoid()
    #     elif activation == 'leaky_relu':
    #         return nn.LeakyReLU()
    #     else:
    #         raise NotImplementedError
        

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x):
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = F.relu(x, inplace=False)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x




class MLP_res(nn.Module):
    """ adapted from https://github.com/CUAI/CorrectAndSmooth/blob/master/gen_models.py """

    def __init__(self, in_channels,out_channels,hidden_channels =64, num_layers = 1, dropout = 1.0):
        super().__init__()

        self.dropout = dropout
        self.lins = nn.ModuleList()
        
        if num_layers == 1:
                self.lins.append(nn.Linear(in_channels, out_channels))
        else:
            self.lins.append(nn.Linear(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.lins.append(nn.Linear(hidden_channels, hidden_channels))

            self.lins.append(nn.Linear(hidden_channels, out_channels))

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x):
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = F.relu(x, inplace=True)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x
    


class MLPgenerator(nn.Module):
    def __init__(self, dim):
        super(MLPgenerator, self).__init__()
        
        def block(in_feat, out_feat, normalize=False, is_last = False):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            if not is_last :
                layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        layer_list = []
        for i, d in enumerate(dim):
            if i == 0:
                layer_list += block(dim[i], dim[i+1], normalize=False)
            elif i < len(dim)-2 :
                layer_list += block(dim[i], dim[i+1])
            elif i == len(dim)-2:
                layer_list += block(dim[i], dim[i+1], is_last = True)
                break
                
        self.model = nn.Sequential(*layer_list)
        
    def forward(self,x):
        
        res = self.model(x) 
        return res