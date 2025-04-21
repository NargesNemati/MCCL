"""IGMC modules"""

import math 
import torch as th 
import torch.nn as nn
import torch.nn.functional as F

from dgl.nn.pytorch import RelGraphConv
from .igmc import IGMC, edge_drop

def uniform(size, tensor):
    bound = 1.0 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-bound, bound)

class IGMC_BERT(IGMC):

    def __init__(self, in_feats, gconv=RelGraphConv, latent_dim=[32, 32, 32, 32], 
                num_relations=5, num_bases=2, regression=False, edge_dropout=0.2, 
                force_undirected=False, side_features=False, n_side_features=0, 
                multiply_by=1):
        super(IGMC, self).__init__()

        self.regression = regression
        self.edge_dropout = edge_dropout
        self.force_undirected = force_undirected
        self.side_features = side_features
        self.multiply_by = multiply_by

        self.convs = th.nn.ModuleList()
        print(in_feats, latent_dim, num_relations, num_bases)
        self.convs.append(gconv(in_feats, latent_dim[0], num_relations, num_bases=num_bases, self_loop=True,))
        for i in range(0, len(latent_dim)-1):
            self.convs.append(gconv(latent_dim[i], latent_dim[i+1], num_relations, num_bases=num_bases, self_loop=True,))
        
        self.lin1 = nn.Linear(2 * sum(latent_dim), 128)
        if side_features:
            self.lin1 = nn.Linear(2 * sum(latent_dim), 64)
            self.lin1_ = nn.Linear(n_side_features, 64)
        if self.regression:
            self.lin2 = nn.Linear(128, 1)
        else:
            assert False
            # self.lin2 = nn.Linear(128, n_classes)
        self.reset_parameters()

    def forward(self, block, bert_vector):
        block = edge_drop(block, self.edge_dropout, self.training)

        concat_states = []
        x = block.ndata['x'].type(th.float32) # one hot feature to emb vector : this part fix errors
        
        for conv in self.convs:
            # edge mask zero denotes the edge dropped
            x = th.tanh(conv(block, x, block.edata['etype'], 
                             norm=block.edata['edge_mask'].unsqueeze(1)))
            concat_states.append(x)
        concat_states = th.cat(concat_states, 1)
        
        users = block.ndata['nlabel'][:, 0] == 1
        items = block.ndata['nlabel'][:, 1] == 1
        # print(bert_vector.shape)
        x = th.cat([concat_states[users], concat_states[items]], 1)
        # print(x.shape)
        bert_vector = F.relu(self.lin1_(bert_vector))
        x = F.relu(self.lin1(x))
        x = th.cat([x, bert_vector], 1)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        if self.regression:
            return x[:, 0] * self.multiply_by
        else:
            assert False