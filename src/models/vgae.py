
from dgl import DropEdge


import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import RelGraphConv
import dgl
from dgl import DropEdge

class GaussianProjection(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim = 32):
        super().__init__()
        # MLP for predicting the mean
        self.mean_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # MLP for predicting the log standard deviation
        self.log_std_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim)
        )


    def forward(self, x):
        mean = self.mean_mlp(x)
        log_std = self.log_std_mlp(x)
        return mean, log_std


class autoencoder(nn.Module):
    
    def __init__(self, in_feats, gconv=RelGraphConv, latent_dim=[32, 32, 32, 32], 
                num_relations=8, num_bases=2, regression=False, edge_dropout=0.2, 
                force_undirected=False, side_features=False, 
                multiply_by=1):
        super(autoencoder, self).__init__()

        self.regression = regression
        self.edge_dropout = 0.05
        self.force_undirected = force_undirected
        self.side_features = side_features
        self.multiply_by = multiply_by
        self.latent_dims = latent_dim
        self.output_vgae = 128

        self.rel_graph_conv_0 = RelGraphConv(in_feats, latent_dim[0], num_relations, 
                                             num_bases=num_bases, self_loop=True)
        self.rel_graph_conv_1 = RelGraphConv(latent_dim[0], latent_dim[1], num_relations, 
                                             num_bases=num_bases, self_loop=True)
        self.rel_graph_conv_2 = RelGraphConv(latent_dim[1], latent_dim[2], num_relations, 
                                             num_bases=num_bases, self_loop=True)
        # self.rel_graph_conv_3 = RelGraphConv(latent_dim[2], latent_dim[3], num_relations, 
        #                                      num_bases=num_bases, self_loop=True)

        self.gaussian_proj = GaussianProjection(3 * latent_dim[0], self.output_vgae)
        self.tanh = th.nn.Tanh()
        self.lin1 = nn.Linear(2 * self.output_vgae, 64)
        self.lin2 = nn.Linear(64, 1)
        self.lin3 = nn.Linear(2 * self.output_vgae, 64)
        self.lin4 = nn.Linear(64, 1)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.lin3.reset_parameters()
        self.lin4.reset_parameters()

    def forward(self, subg):

        src = subg.edata['original_src_idx']
        dst = subg.edata['original_dst_idx']

        target_edges_mask = subg.edata['is_target_edge'] == 1

        node_ids = subg.ndata['node_id']
        index_map = {}
        for index, value in enumerate(node_ids):
            if value not in index_map:  
                index_map[value.item()] = index

        src_indices = [index_map[value] if value in index_map else 1 for value in src.tolist()]
        dst_indices = [index_map[value] if value in index_map else 1 for value in dst.tolist()]
        if self.training:
            drop_edge_transform = DropEdge(p=self.edge_dropout)
            subg = drop_edge_transform(subg)
        subg = dgl.add_self_loop(subg)

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
        concat_states = th.cat(concat_states, 1)

        self.mean, self.log_std = self.gaussian_proj(concat_states)

        gaussian_noise = th.randn(self.mean.shape).to(0)

        z = self.mean + gaussian_noise * th.exp(self.log_std).to(0)



        users = subg.ndata['nlabel'][:, 0] == 1
        items = subg.ndata['nlabel'][:, 1] == 1
        results = th.cat([z[users], z[items]], dim=0)

        users = z[users]
        items = z[items]


        x = (users * items).sum(dim=1)


        if self.training:
            src_emb = z[src_indices]
            dst_emb = z[dst_indices]
            preds = (src_emb * dst_emb).sum(dim=1)


        else:
            edge_masks = 0
            preds = 0
        if self.regression:
            return x, preds, results
            assert False

    def __repr__(self):
        return self.__class__.__name__
    