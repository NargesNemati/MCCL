
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from dgl import DropEdge
from torch import nn
from dgl import function as fn
from dgl.nn import TypedLinear


class RelGraphConv(nn.Module):

    def __init__(self,
                 in_feat,
                 out_feat,
                 num_rels,
                 regularizer=None,
                 num_bases=None,
                 bias=True,
                 activation=None,
                 self_loop=True,
                 dropout=0.0,
                 layer_norm=False):
        super().__init__()
        self.linear_r = TypedLinear(in_feat, out_feat, num_rels, regularizer, num_bases)
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop
        self.layer_norm = layer_norm

        # bias
        if self.bias:
            self.h_bias = nn.Parameter(th.Tensor(out_feat))
            nn.init.zeros_(self.h_bias)

        # TODO(minjie): consider remove those options in the future to make
        #   the module only about graph convolution.
        # layer norm
        if self.layer_norm:
            self.layer_norm_weight = nn.LayerNorm(out_feat, elementwise_affine=True)

        # weight for self loop
        if self.self_loop:
            self.loop_weight = nn.Parameter(th.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.loop_weight, gain=nn.init.calculate_gain('relu'))

        self.dropout = nn.Dropout(dropout)

    def message(self, edges):
        """Message function."""
        m = self.linear_r(edges.src['h'], edges.data['etype'], self.presorted)
        if 'norm' in edges.data:
            m = m * edges.data['norm']
        if 'w' in edges.data:  # Apply additional edge weights
            m = m * edges.data['w'].unsqueeze(1)
        return {'m': m}


    def forward(self, g, feat, etypes, norm=None, *, presorted=False):

        self.presorted = presorted
        with g.local_scope():
            g.srcdata['h'] = feat
            if norm is not None:
                g.edata['norm'] = norm
            g.edata['etype'] = etypes
            # message passing
            g.update_all(self.message, fn.sum('m', 'h'))
            # apply bias and activation
            h = g.dstdata['h']
            if self.layer_norm:
                h = self.layer_norm_weight(h)
            if self.bias:
                h = h + self.h_bias
            if self.self_loop:
                h = h + feat[:g.num_dst_nodes()] @ self.loop_weight
            if self.activation:
                h = self.activation(h)
            h = self.dropout(h)
            return h


class EdgeClassifier(nn.Module):
    def __init__(self, input_dim):
        super(EdgeClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.fc(x)
    
class KGMC_att(nn.Module):
    # RGCN convolution
    
    def __init__(self, in_feats, gconv=RelGraphConv, latent_dim=[32, 32, 32, 32], 
                num_relations=8, num_bases=2, regression=False, edge_dropout=0.2, 
                force_undirected=False, side_features=False, 
                multiply_by=1):
        super(KGMC_att, self).__init__()

        self.regression = regression
        self.edge_dropout = edge_dropout
        self.force_undirected = force_undirected
        self.side_features = side_features
        self.multiply_by = multiply_by
        self.latent_dims = latent_dim

        self.attentions_0 = nn.Linear(2* latent_dim[0], 1)
        self.attentions_1 = nn.Linear(2* latent_dim[1], 1)
        self.attentions_2 = nn.Linear(2* latent_dim[2], 1)

        self.rel_graph_conv_0 = RelGraphConv(in_feats, latent_dim[0], num_relations, 
                                             num_bases=num_bases, self_loop=True)
        self.rel_graph_conv_1 = RelGraphConv(latent_dim[0], latent_dim[1], num_relations, 
                                             num_bases=num_bases, self_loop=True)
        self.rel_graph_conv_2 = RelGraphConv(latent_dim[1], latent_dim[2], num_relations, 
                                             num_bases=num_bases, self_loop=True)
        self.rel_graph_conv_3 = RelGraphConv(latent_dim[2], latent_dim[3], num_relations, 
                                             num_bases=num_bases, self_loop=True)
        
        self.classifier_0 = EdgeClassifier(1)
        self.classifier_1 = EdgeClassifier(2* latent_dim[0])
        self.classifier_2 = EdgeClassifier(2* latent_dim[0])
        self.classifier_3 = EdgeClassifier(2* latent_dim[0])

        self.mean_layer = nn.Linear(sum(self.latent_dims) , 128)  # Outputs mean
        self.log_std_layer = nn.Linear(sum(self.latent_dims) , 128)  # Outputs log(std)
        self.tanh = th.nn.Tanh()
        self.lin1 = nn.Linear(2 * sum(self.latent_dims), 128)

        if self.regression:
            self.lin2 = nn.Linear(2 * sum(self.latent_dims), 1)
        else:
            assert False
        self.reset_parameters()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, subg):
        if self.training:
            drop_edge_transform = DropEdge(p=self.edge_dropout)
            subg = drop_edge_transform(subg)

        src = subg.edata['original_src_idx']
        dst = subg.edata['original_dst_idx']
        node_ids = subg.ndata['node_id']


        index_map = {}
        for index, value in enumerate(node_ids):
            if value not in index_map: 
                index_map[value.item()] = index


        src_indices = [index_map[value] if value in index_map else 1 for value in src.tolist()]
        dst_indices = [index_map[value] if value in index_map else 1 for value in dst.tolist()]
        all_edge_mask = []

        concat_states = []
        x = subg.ndata['x'].type(th.float32)
        e = subg.edata['etype']
        e = e.to(th.float32).clone().detach().to(0)

        x = self.rel_graph_conv_0(subg, x, e)
        x = self.tanh(x)
        concat_states.append(x)

        temp_concat_states = th.cat(concat_states)
        all_edges = th.cat([temp_concat_states[src_indices], temp_concat_states[dst_indices]], 1)
        edges_weights = self.classifier_2(all_edges).squeeze()
        edges_for_del = (edges_weights - edges_weights.min()) / (edges_weights.max() - edges_weights.min() + 1e-8)
        subg.edata['w'] = edges_for_del

        all_edge_mask.append(edges_for_del)

        x = self.rel_graph_conv_1(subg, x, e)
        x = self.tanh(x)
        concat_states.append(x)

        temp_concat_states = th.cat(concat_states)
        all_edges = th.cat([temp_concat_states[src_indices], temp_concat_states[dst_indices]], 1)
        edges_weights = self.classifier_2(all_edges).squeeze()
        edges_for_del = (edges_weights - edges_weights.min()) / (edges_weights.max() - edges_weights.min() + 1e-8)
        subg.edata['w'] = edges_for_del
        all_edge_mask.append(edges_for_del)

        x = self.rel_graph_conv_2(subg, x, e)
        x = self.tanh(x)
        concat_states.append(x)

        temp_concat_states = th.cat(concat_states)
        all_edges = th.cat([temp_concat_states[src_indices], temp_concat_states[dst_indices]], 1)
        edges_for_del = (edges_weights - edges_weights.min()) / (edges_weights.max() - edges_weights.min() + 1e-8)
        subg.edata['w'] = edges_for_del
        all_edge_mask.append(edges_for_del)

        x = self.rel_graph_conv_3(subg, x, e)
        x = self.tanh(x)
        concat_states.append(x)

        concat_states = th.cat(concat_states, 1)

        users = subg.ndata['nlabel'][:, 0] == 1
        items = subg.ndata['nlabel'][:, 1] == 1

        users = x[users]
        items = x[items]

        emb = th.cat([users, items], dim=0)
        x = (users * items).sum(dim=1)
        return x, all_edge_mask, concat_states


    def __repr__(self):
        return self.__class__.__name__
    
def edge_drop_classification(graph, edge_drop,  src, dst, training=True):
    g = graph
    binary_tensor = ((edge_drop > 0.5).int()).to(0)
    edge_ids = th.arange(g.num_edges(), device=g.device, dtype=th.int32)
    binary_tensor = (edge_drop > 0.5).int()
    edges_to_remove = edge_ids[binary_tensor == 0]
    g = dgl.remove_edges(g, edges_to_remove)
    edge_ids = th.arange(g.num_edges(), device=g.device, dtype=th.int32)
    return g
