from dgl import DropEdge
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl import DropEdge
from dgl.nn.pytorch import RelGraphConv, NNConv, EGATConv
from dgl.nn import RelGraphConv

class KGMC(nn.Module):

    def __init__(self, in_feats, gconv=RelGraphConv, latent_dim=[32, 32, 32, 32], 
                num_relations=8, num_bases=2, regression=False, edge_dropout=0.2, 
                force_undirected=False, side_features=False, 
                multiply_by=1):
        super(KGMC, self).__init__()

        self.regression = regression
        self.edge_dropout = edge_dropout
        self.force_undirected = force_undirected
        self.side_features = side_features
        self.multiply_by = multiply_by

        self.rel_graph_conv_0 = RelGraphConv(in_feats, latent_dim[0], num_relations, 
                                             num_bases=num_bases, self_loop=True)
        self.rel_graph_conv_1 = RelGraphConv(latent_dim[0], latent_dim[1], num_relations, 
                                             num_bases=num_bases, self_loop=True)
        self.rel_graph_conv_2 = RelGraphConv(latent_dim[1], latent_dim[2], num_relations, 
                                             num_bases=num_bases, self_loop=True)
        self.rel_graph_conv_3 = RelGraphConv(latent_dim[2], latent_dim[3], num_relations, 
                                             num_bases=num_bases, self_loop=True)

        self.tanh = th.nn.Tanh()
        self.lin1 = nn.Linear(2 * sum(latent_dim), 1)
        if self.regression:
            self.lin2 = nn.Linear(128, 1)
        else:
            assert False
        self.reset_parameters()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, subg):
        # subg = edge_drop(subg, self.edge_dropout, self.training)
        if self.training:
            drop_edge_transform = DropEdge(p=self.edge_dropout)
            subg = drop_edge_transform(subg)

        concat_states = []
        x = subg.ndata['x'].type(th.float32)
        e = subg.edata['etype']

        x = self.rel_graph_conv_0(subg, x, e)
        x = self.tanh(x)
        concat_states.append(x)

        x = self.rel_graph_conv_1(subg, x, e)
        x = self.tanh(x)
        concat_states.append(x)

        x = self.rel_graph_conv_2(subg, x, e)
        x = self.tanh(x)
        concat_states.append(x)

        x = self.rel_graph_conv_3(subg, x, e)
        x = self.tanh(x)
        concat_states.append(x)
        concat_states = th.cat(concat_states, 1)

        users = subg.ndata['nlabel'][:, 0] == 1
        items = subg.ndata['nlabel'][:, 1] == 1
        x = th.cat([concat_states[users], concat_states[items]], 1)

        # x = self.lin1(x)
        # x = F.dropout(x, p=0.5, training=self.training)
        # x = th.sigmoid(self.lin1(x))
        # x = F.relu(self.lin1(x))
        # x = F.dropout(x, p=0.5, training=self.training)
        x = th.sigmoid(self.lin1(x))
        # x = F.dropout(x, p=0.5, training=self.training)
        # x = th.sigmoid(self.lin1(x))
        if self.regression:
            return x[:, 0] * self.multiply_by, 0, 0
        else:
            assert False

    def __repr__(self):
        return self.__class__.__name__


def edge_drop(graph, edge_dropout=0.2, training=True):
    assert edge_dropout >= 0.0 and edge_dropout <= 1.0, 'Invalid dropout rate.'
    if not training:
        return graph
    src, _ = graph.edges()
    to_drop = src.new_full((graph.number_of_edges(),), edge_dropout, dtype=th.float)
    to_drop = th.bernoulli(to_drop).to(th.bool)
    graph.edata['edge_mask'][to_drop] = 0
    if 'etype' in graph.edata:
        graph.edata['etype'] = graph.edata['etype']
    return graph



