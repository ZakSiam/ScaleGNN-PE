import torch.nn as nn
from graph_env.graph_generator import Graph
from config import *

class NN(nn.Module):
    def __init__(self, input_dim, depth, width, aggr_feats = False):
        super(NN, self).__init__()
        self.aggr_feats = aggr_feats
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim,width))
        self.layers.append(nn.ReLU())
        for i in range(depth-1):
            self.layers.append(nn.Linear(width, width))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(width,1))


    def forward(self, g: Graph):
        dev = next(self.parameters()).device
        if self.aggr_feats:
            feat_mat = torch.from_numpy(g.feat_mat_aggr_normed()).float().to(dev)
        else:
            feat_mat = torch.from_numpy(g.feat_mat_normed()).float().to(dev)

        x = torch.flatten(feat_mat)
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        return x

class GNN(nn.Module):
    def __init__(self, input_dim:int, depth:int, width: int, aggr_feats: bool = True):
        super(GNN, self).__init__()
        self.aggr_feats = aggr_feats
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim,width))
        self.layers.append(nn.ReLU())
        for i in range(depth - 1):
            self.layers.append(nn.Linear(width, width))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(width,1))

    def forward(self, g: Graph):
        dev = next(self.parameters()).device
        if self.aggr_feats:
            x = torch.from_numpy(g.feat_mat_aggr_normed()).float().to(dev)
        else:
            x = torch.from_numpy(g.feat_mat_normed()).float().to(dev)

        for i in range(len(self.layers)):
            x = self.layers[i](x)
        return torch.mean(x)



class GNNEmbedder(nn.Module):
    """Plain (fixed) GNN embedder: graph -> vector.

    This is intentionally simple: a stack of Linear+ReLU applied to node features
    (either aggregated features or raw features), followed by mean pooling across nodes.

    The output is a vector of size `width`.
    """
    def __init__(self, input_dim: int, depth: int, width: int, aggr_feats: bool = True):
        super(GNNEmbedder, self).__init__()
        self.aggr_feats = aggr_feats
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, width))
        self.layers.append(nn.ReLU())
        for _ in range(depth - 1):
            self.layers.append(nn.Linear(width, width))
            self.layers.append(nn.ReLU())

    def forward(self, g: Graph):
        dev = next(self.parameters()).device
        if self.aggr_feats:
            x = torch.from_numpy(g.feat_mat_aggr_normed()).float().to(dev)
        else:
            x = torch.from_numpy(g.feat_mat_normed()).float().to(dev)

        for layer in self.layers:
            x = layer(x)  # [num_nodes, width]
        # mean pool -> [width]
        return torch.mean(x, dim=0)

def normalize_init(net):
    '''
    :param net: input network, random state
    :return: network that is normalized wrt xavier initialization
    '''
    layers_list = [module for module in net.modules() if type(module) == nn.Linear]
    for layer in layers_list[1:]:
        #nn.init.xavier_normal_(layer.weight,gain=np.sqrt(2))
        nn.init.kaiming_normal_(layer.weight, mode = 'fan_in', nonlinearity = 'relu')
        #layer.bias.data.fill_(0.0)
    return net