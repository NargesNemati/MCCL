import torch as th
import torch.nn as nn
import torch.nn.functional as F
# from dgl.nn import RelGraphConv

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl import DGLGraph
import dgl.function as fn
from functools import partial

from dgl import DropEdge



# pylint: disable= no-member, arguments-differ, invalid-name
import torch as th
from torch import nn

from dgl import function as fn
from dgl.nn import TypedLinear

# class contrastive(nn.Module):
#     def __init__(self, input_dim):
#         super(contrastive, self).__init__()
#         self.fc_user = nn.Sequential(
#             nn.Linear(input_dim, 64),
#             nn.ReLU(),
#             nn.Linear(64, 64),
#             nn.Sigmoid()
#         )
#         self.fc_item = nn.Sequential(
#             nn.Linear(input_dim, 64),
#             nn.ReLU(),
#             nn.Linear(64, 64),
#             nn.Sigmoid()
#         )
#         self.fc_edges = nn.Sequential(
#             nn.Linear(128, 64),
#             nn.ReLU(),
#             nn.Linear(64, 1),
#             nn.Sigmoid()
#         )

class contrastive(nn.Module):
    def __init__(self, input_dim):
        super(contrastive, self).__init__()
        self.fc_user = nn.Sequential(
            nn.Linear(input_dim, 64),
            # nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 64),
            # nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5)
        )
        self.fc_item = nn.Sequential(
            nn.Linear(input_dim, 64),
            # nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 64),
            # nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5)
        )
        # self.fc_edges = nn.Sequential(
        #     nn.Linear(128, 64),
        #     # nn.BatchNorm1d(64),
        #     nn.LeakyReLU(0.2),
        #     nn.Linear(64, 1),
        #     nn.Sigmoid()
        # )    
        self.lin1 = nn.Linear(128, 64)

        self.lin2 = nn.Linear(64, 1)
    def forward(self, emb1, emb2, subg, temp=0.5):
        emb1 = emb1.to(0)
        emb2 = emb2.to(0)
        x = th.cat([emb1, emb2], dim=1).to(0)
        users = subg.ndata['nlabel'][:, 0] == 1
        items = subg.ndata['nlabel'][:, 1] == 1
        # users = self.fc_user(x[users])
        # items = self.fc_item(x[items])
        # edges = th.cat([users, items], 1).to(0)
        # preds = self.fc_edges(edges)
        users = x[users]
        items = x[items]
        preds = (users * items).sum(dim=1)  # shape: (batch_size,)
        preds = th.sigmoid(preds)
        return preds

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLearner(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        # self.projection = nn.Sequential(
        #     nn.Linear(embedding_dim, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 64)
        # )
        # self.contrast = contrastive(2 * embedding_dim)
        self.fc_user = nn.Sequential(
            nn.Linear(2 * embedding_dim, 64),
            # nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 64),
            # nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5)
        )
        self.fc_item = nn.Sequential(
            nn.Linear(2 * embedding_dim, 64),
            # nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 64),
            # nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5)
        )
        self.fc_edges = nn.Sequential(
            nn.Linear(512, 64),
            # nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        # self.lin1 = nn.Linear(512, 64)

        # self.lin2 = nn.Linear(64, 1)

    def forward(self, emb1, emb2, subg, tau):
    #     # Identify user and item nodes
    #     users = subg.ndata['nlabel'][:, 0] == 1
    #     items = subg.ndata['nlabel'][:, 1] == 1

    #     # Project embeddings
    #     z1 = self.projection(emb1)
    #     z2 = self.projection(emb2)

    #     # Normalize embeddings (optional)
    #     # z1 = F.normalize(z1, dim=1)
    #     # z2 = F.normalize(z2, dim=1)

    #     # Split into user and item embeddings
    #     user_z1 = z1[users]
    #     user_z2 = z2[users]
    #     item_z1 = z1[items]
    #     item_z2 = z2[items]

        # Identify user and item nodes
        users = subg.ndata['nlabel'][:, 0] == 1
        items = subg.ndata['nlabel'][:, 1] == 1

        z_1 = emb2[users]
        z_2 = emb2[items]
        emb2 = th.cat([z_1, z_2], dim=0)
        len_target = int(len(emb2)/2)
        # Split into user and item embeddings
        # z1 = self.projection(emb1)
        # z2 = self.projection(emb2)

        user_z1 = emb1[:len_target]
        user_z2 = emb2[:len_target]
        item_z1 = emb1[len_target:]
        item_z2 = emb2[len_target:]


        user_z1 = F.normalize(user_z1, dim=1)
        user_z2 = F.normalize(user_z2, dim=1)
        item_z1 = F.normalize(item_z1, dim=1)
        item_z2 = F.normalize(item_z2, dim=1)
        # print(item_z1)

        # Helper function to compute contrastive loss
        def compute_loss(z1_split, z2_split, tau):
            # if z1_split.size(0) == 0:  # Handle empty tensors
            #     return torch.tensor(0.0, device=z1_split.device)
            similarities = torch.mm(z1_split, z2_split.T) / tau
            n = similarities.size(0)
            loss = 0.0
            for i in range(n):
                pos_sim = similarities[i, i]
                # print(pos_sim)
                neg_sim = torch.cat([similarities[i, :i], similarities[i, i+1:]])
                denom = torch.sum(torch.exp(neg_sim))
                loss -= torch.log(torch.exp(pos_sim) / (denom + 1e-8))  # Add epsilon to avoid log(0)

            # return loss / n if n > 0 else torch.tensor(0.0, device=z1_split.device)
            return loss / n

        # Compute losses for users and items
        user_loss = compute_loss(user_z2, user_z1, tau)
        item_loss = compute_loss(item_z2, item_z1, tau)

        # Combine losses
        total_loss = (user_loss + item_loss) / 2
        # print(emb2)

        # Compute predictions (unchanged)
        # preds = self.contrast(emb1, emb2, subg)
        x = th.cat([emb1, emb2], dim=1).to(0)
        
        users = x[:len_target]
        items = x[len_target:]
        # users = self.fc_user(x[:len_target])
        # items = self.fc_item(x[len_target:])
        edges = th.cat([users, items], 1).to(0)
        # edges = F.relu(self.lin1(edges))
        # edges = F.dropout(edges, p=0.5, training=self.training)
        # preds = th.sigmoid(self.lin2(edges))
        # preds = preds[:, 0]
        preds = self.fc_edges(edges)
        preds = preds.squeeze(1)
        # users_norm = F.normalize(users, dim=1)  # Shape: (batch_size, d)
        # items_norm = F.normalize(items, dim=1)
        # preds = (users * items).sum(dim=1)  # shape: (batch_size,)
        # preds = th.sigmoid(preds)
        # x = F.softmax(preds, dim=1)

        return total_loss, preds

# class ContrastiveLearner(nn.Module):
#     def __init__(self, embedding_dim):
#         super().__init__()
#         self.projection = nn.Sequential(
#             nn.Linear(embedding_dim, 64),
#             nn.ReLU(),
#             nn.Linear(64, 64)
#         )

#         self.contrast = contrastive(2 * embedding_dim)

#     def forward(self, emb1, emb2, subg, tau):
#         # Identify user and item nodes
#         users = subg.ndata['nlabel'][:, 0] == 1
#         items = subg.ndata['nlabel'][:, 1] == 1
        
#         # Generate positive pairs: (k, k) for all users and items
#         user_indices = torch.where(users)[0].tolist()
#         item_indices = torch.where(items)[0].tolist()
#         pos_pairs = [(i, i) for i in user_indices] + [(j, j) for j in item_indices]

#         # Project embeddings
#         z1 = self.projection(emb1)
#         z2 = self.projection(emb2)

#         # Normalize embeddings
#         z1 = F.normalize(z1, dim=1)
#         z2 = F.normalize(z2, dim=1)

#         # Compute contrastive loss
#         loss = 0
#         for (i, j) in pos_pairs:
#             anchor = z1[i]
#             positive = z2[j]

#             # Positive similarity
#             pos_sim = torch.dot(anchor, positive)

#             # Negative similarities (all nodes except j)
#             neg_sim = torch.mm(anchor.unsqueeze(0), z2.T).squeeze(0)
#             mask = torch.ones_like(neg_sim, dtype=torch.bool)
#             mask[j] = False
#             neg_sim = neg_sim[mask]

#             # Skip if no negatives
#             if len(neg_sim) == 0:
#                 continue

#             # InfoNCE loss
#             logits = torch.cat([pos_sim.unsqueeze(0), neg_sim])
#             labels = torch.zeros(1, dtype=torch.long).to(logits.device)
#             loss += F.cross_entropy(logits.unsqueeze(0) / tau, labels)
#             loss = loss / len(pos_pairs)

#             preds = self.contrast(emb1.detach(), emb2.detach(), subg)

#         return loss, preds


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
        # subg = edge_drop(subg, self.edge_dropout, self.training)
        # new_subg = subg
        # edge_dropout = 0.1 * min(epoch_idx, 1)

        # print(edge_dropout)
        if self.training:
            drop_edge_transform = DropEdge(p=self.edge_dropout)
            subg = drop_edge_transform(subg)

        #-------------------------------s---------------
        src = subg.edata['original_src_idx']
        dst = subg.edata['original_dst_idx']
        node_ids = subg.ndata['node_id']

        # true_tensor = th.ones(len(src), device='cuda:0', dtype=th.bool)  # Boolean tensor
        index_map = {}
        for index, value in enumerate(node_ids):
            if value not in index_map:  # Only add the first occurrence
                index_map[value.item()] = index


        src_indices = [index_map[value] if value in index_map else 1 for value in src.tolist()]
        dst_indices = [index_map[value] if value in index_map else 1 for value in dst.tolist()]
        all_edge_mask = []
        #-------------------------------e--------------

        concat_states = []
        x = subg.ndata['x'].type(th.float32)
        e = subg.edata['etype']  # Edge type (relations)
        e = e.to(th.float32).clone().detach().to(0)

        # Apply tanh activation function after each layer
        x = self.rel_graph_conv_0(subg, x, e)
        x = self.tanh(x)
        concat_states.append(x)

        temp_concat_states = th.cat(concat_states)
        all_edges = th.cat([temp_concat_states[src_indices], temp_concat_states[dst_indices]], 1)
        edges_weights = self.classifier_2(all_edges).squeeze()
        edges_for_del = (edges_weights - edges_weights.min()) / (edges_weights.max() - edges_weights.min() + 1e-8)
        subg.edata['w'] = edges_for_del
        # new_subg = edge_drop_classification(subg, edges_for_del, src, dst)
        # e = new_subg.edata['etype']
        all_edge_mask.append(edges_for_del)

        x = self.rel_graph_conv_1(subg, x, e)
        x = self.tanh(x)
        concat_states.append(x)

        temp_concat_states = th.cat(concat_states)
        all_edges = th.cat([temp_concat_states[src_indices], temp_concat_states[dst_indices]], 1)
        edges_weights = self.classifier_2(all_edges).squeeze()
        # print(edges_weights)
        edges_for_del = (edges_weights - edges_weights.min()) / (edges_weights.max() - edges_weights.min() + 1e-8)
        subg.edata['w'] = edges_for_del
        # new_subg = edge_drop_classification(subg, edges_for_del, src, dst)
        # e = new_subg.edata['etype']
        all_edge_mask.append(edges_for_del)

        x = self.rel_graph_conv_2(subg, x, e)
        x = self.tanh(x)
        concat_states.append(x)

        temp_concat_states = th.cat(concat_states)
        all_edges = th.cat([temp_concat_states[src_indices], temp_concat_states[dst_indices]], 1)
        edges_for_del = (edges_weights - edges_weights.min()) / (edges_weights.max() - edges_weights.min() + 1e-8)
        subg.edata['w'] = edges_for_del
        # new_subg = edge_drop_classification(subg, edges_for_del, src, dst)
        # new_subg = edge_drop_classification(subg, edges_for_del, src, dst)
        # e = new_subg.edata['etype']
        all_edge_mask.append(edges_for_del)

        x = self.rel_graph_conv_3(subg, x, e)
        x = self.tanh(x)
        concat_states.append(x)

        concat_states = th.cat(concat_states, 1)

        users = subg.ndata['nlabel'][:, 0] == 1
        items = subg.ndata['nlabel'][:, 1] == 1
        x = th.cat([concat_states[users], concat_states[items]], 1)
        
        # x = F.dropout(x, p=0.5, training=self.training)
        # x = th.sigmoid(self.lin2(x))

        # x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = th.sigmoid(self.lin2(x))

        return x[:, 0] * self.multiply_by, all_edge_mask

    def __repr__(self):
        return self.__class__.__name__
    



class KGMC_autoencoder(nn.Module):
    # RGCN convolution
    
    def __init__(self, in_feats, gconv=RelGraphConv, latent_dim=[32, 32, 32, 32], 
                num_relations=8, num_bases=2, regression=False, edge_dropout=0.2, 
                force_undirected=False, side_features=False, 
                multiply_by=1):
        super(KGMC_autoencoder, self).__init__()

        self.regression = regression
        self.edge_dropout = 0.05
        self.force_undirected = force_undirected
        self.side_features = side_features
        self.multiply_by = multiply_by
        self.latent_dims = latent_dim
        self.output_vgae = 32

        self.rel_graph_conv_0 = RelGraphConv(in_feats, latent_dim[0], num_relations, 
                                             num_bases=num_bases, self_loop=True)
        self.rel_graph_conv_1 = RelGraphConv(latent_dim[0], latent_dim[1], num_relations, 
                                             num_bases=num_bases, self_loop=True)
        self.rel_graph_conv_2 = RelGraphConv(latent_dim[1], latent_dim[2], num_relations, 
                                             num_bases=num_bases, self_loop=True)

        self.gaussian_proj = GaussianProjection(3 * latent_dim[0], self.output_vgae)
        self.tanh = th.nn.Tanh()
        self.lin1 = nn.Linear(2 * self.output_vgae, 64)
        self.lin2 = nn.Linear(64, 1)
        self.lin3 = nn.Linear(2 * self.output_vgae, 64)
        self.lin4 = nn.Linear(64, 1)


        # if self.regression:
        #     self.lin2 = nn.Linear(4 * latent_dim[0], 1)
        # else:
        #     assert False
        self.reset_parameters()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.lin3.reset_parameters()
        self.lin4.reset_parameters()

    def forward(self, subg):

        src = subg.edata['original_src_idx']
        # print(src_edge)
        dst = subg.edata['original_dst_idx']

        target_edges_mask = subg.edata['is_target_edge'] == 1

        node_ids = subg.ndata['node_id']
        # subg = edge_drop(subg, self.edge_dropout, self.training)
        index_map = {}
        for index, value in enumerate(node_ids):
            if value not in index_map:  # Only add the first occurrence
                index_map[value.item()] = index

        src_indices = [index_map[value] if value in index_map else 1 for value in src.tolist()]
        dst_indices = [index_map[value] if value in index_map else 1 for value in dst.tolist()]
        # print(len(src_indices))
        if self.training:
            drop_edge_transform = DropEdge(p=self.edge_dropout)
            subg = drop_edge_transform(subg)
        subg = dgl.add_self_loop(subg)

        concat_states = []
        x = subg.ndata['x'].type(th.float32)
        e = subg.edata['etype']  # Edge type (relations)


        # Apply tanh activation function after each layer
        x = self.rel_graph_conv_0(subg, x, e)
        x = self.tanh(x)
        concat_states.append(x)

        x = self.rel_graph_conv_1(subg, x, e)
        x = self.tanh(x)
        concat_states.append(x)

        x = self.rel_graph_conv_2(subg, x, e)
        x = self.tanh(x)
        concat_states.append(x)


        # # Concatenate states from all layers
        concat_states = th.cat(concat_states, 1)
        users = subg.ndata['nlabel'][:, 0] == 1
        items = subg.ndata['nlabel'][:, 1] == 1
        
        results = th.cat([concat_states[users], concat_states[items]], dim=0)

        # self.mean = self.mean_layer(x)
        # self.log_std = self.log_std_layer(x)
        self.mean, self.log_std = self.gaussian_proj(results)

        gaussian_noise = th.randn(self.mean.shape).to(0)

        z = self.mean + gaussian_noise * th.exp(self.log_std).to(0)

        # src_emb = z[src_indices]
        # dst_emb = z[dst_indices]


        # users = subg.ndata['nlabel'][:, 0] == 1
        # items = subg.ndata['nlabel'][:, 1] == 1
        # x = th.cat([z[users], z[items]], 1)


        len_target = int(len(z)/2)
        z_1 = z[:len_target]
        z_2 = z[len_target:]
        x = th.cat([z_1, z_2], 1)
        
        # x = F.dropout(x, p=0.5, training=self.training)
        # x = th.sigmoid(self.lin1(x))
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = th.sigmoid(self.lin2(x))

        if self.training:
            # preds = th.cat([z[src_indices], z[dst_indices]], 1)
            # # preds= F.dropout(preds, p=0.5, training=self.training)
            # # preds = th.sigmoid(self.lin3(preds))
            # preds = F.relu(self.lin3(preds))
            # preds = F.dropout(preds, p=0.5, training=self.training)
            # preds = th.sigmoid(self.lin4(preds))
            # # preds = self.lin2(preds)
            # preds = preds[:, 0] * self.multiply_by
            preds = 0 

        else:
            edge_masks = 0
            preds = 0
        if self.regression:
            return x[:, 0] * self.multiply_by, preds
            assert False

    def __repr__(self):
        return self.__class__.__name__
    

from dgl.nn import SAGEConv

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch.conv import SAGEConv
from dgl import DropEdge

class KGMC_SAGE(nn.Module):
    def __init__(self, in_feats, latent_dim=[32, 32, 32, 32], 
                 regression=False, edge_dropout=0.2, multiply_by=1):
        super(KGMC_SAGE, self).__init__()
        self.regression = regression
        self.edge_dropout = edge_dropout
        self.multiply_by = multiply_by

        # Replace RelGraphConv layers with SAGEConv layers (using 'mean' aggregator)
        self.sage_conv_0 = SAGEConv(in_feats, latent_dim[0], aggregator_type='mean')
        self.sage_conv_1 = SAGEConv(latent_dim[0], latent_dim[1], aggregator_type='mean')
        self.sage_conv_2 = SAGEConv(latent_dim[1], latent_dim[2], aggregator_type='mean')
        self.sage_conv_3 = SAGEConv(latent_dim[2], latent_dim[3], aggregator_type='mean')
        
        self.tanh = nn.Tanh()
        # The linear layers remain unchanged
        self.lin1 = nn.Linear(2 * sum(latent_dim), 128)
        if self.regression:
            self.lin2 = nn.Linear(sum(latent_dim), 1)
        else:
            assert False, "Regression mode must be True."
        self.reset_parameters()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, subg):
        # Optionally apply edge dropout if training.
        if self.training:
            drop_edge_transform = DropEdge(p=self.edge_dropout)
            subg = drop_edge_transform(subg)

        concat_states = []
        # Extract node features as float
        x = subg.ndata['x'].float()

        # Process through each SAGEConv layer with tanh activation.
        x = self.sage_conv_0(subg, x)
        x = self.tanh(x)
        concat_states.append(x)

        x = self.sage_conv_1(subg, x)
        x = self.tanh(x)
        concat_states.append(x)

        x = self.sage_conv_2(subg, x)
        x = self.tanh(x)
        concat_states.append(x)

        x = self.sage_conv_3(subg, x)
        x = self.tanh(x)
        concat_states.append(x)

        # Concatenate the outputs from all layers along feature dimension.
        concat_states = th.cat(concat_states, dim=1)

        # Assuming that subg.ndata['nlabel'] contains two columns identifying user/item nodes.
        users = subg.ndata['nlabel'][:, 0] == 1
        items = subg.ndata['nlabel'][:, 1] == 1
        x = th.cat([concat_states[users], concat_states[items]], dim=1)

        # Fully connected layers and dropout.
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = th.sigmoid(self.lin2(x))
        if self.regression:
            return x[:, 0] * self.multiply_by, 0, concat_states
        else:
            assert False, "Regression mode must be True."

    def __repr__(self):
        return self.__class__.__name__


    



