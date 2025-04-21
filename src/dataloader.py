from ast import keyword
from copy import copy
from re import sub
import torch as th
import pandas as pd
import numpy as np
import dgl
from queue import PriorityQueue

from itertools import combinations
from user_item_graph import UserItemGraph

#######################
# Subgraph Extraction 
#######################
def one_hot(idx, length):
    x = th.zeros([len(idx), length], dtype=th.int32)
    x[th.arange(len(idx)), idx] = 1.0
    return x


def get_neighbor_nodes_labels(u_node_idx, i_node_idx, graph, 
                              hop=1, sample_ratio=1.0, max_nodes_per_hop=200):

    # 1. neighbor nodes sampling
    dist = 0
    u_nodes, i_nodes = th.unsqueeze(u_node_idx, 0), th.unsqueeze(i_node_idx, 0)
    u_dist, i_dist = th.tensor([0], dtype=th.long), th.tensor([0], dtype=th.long)
    u_visited, i_visited = th.unique(u_nodes), th.unique(i_nodes)
    u_fringe, i_fringe = th.unique(u_nodes), th.unique(i_nodes)

    for dist in range(1, hop+1):
        # sample neigh alternately
        # diff from original code : only use one-way edge (u-->i)
        u_fringe, i_fringe = graph.in_edges(i_fringe)[0], graph.out_edges(u_fringe)[1]
        u_fringe = th.from_numpy(np.setdiff1d(u_fringe.numpy(), u_visited.numpy()))
        i_fringe = th.from_numpy(np.setdiff1d(i_fringe.numpy(), i_visited.numpy()))
        u_visited = th.unique(th.cat([u_visited, u_fringe]))
        i_visited = th.unique(th.cat([i_visited, i_fringe]))

        if sample_ratio < 1.0:
            shuffled_idx = th.randperm(len(u_fringe))
            u_fringe = u_fringe[shuffled_idx[:int(sample_ratio*len(u_fringe))]]
            shuffled_idx = th.randperm(len(i_fringe))
            i_fringe = i_fringe[shuffled_idx[:int(sample_ratio*len(i_fringe))]]
        if max_nodes_per_hop is not None:
            if max_nodes_per_hop < len(u_fringe):
                shuffled_idx = th.randperm(len(u_fringe))
                u_fringe = u_fringe[shuffled_idx[:max_nodes_per_hop]]
            if max_nodes_per_hop < len(i_fringe):
                shuffled_idx = th.randperm(len(i_fringe))
                i_fringe = i_fringe[shuffled_idx[:max_nodes_per_hop]]
        if len(u_fringe) == 0 and len(i_fringe) == 0:
            break

        u_nodes = th.cat([u_nodes, u_fringe])
        i_nodes = th.cat([i_nodes, i_fringe])
        u_dist = th.cat([u_dist, th.full((len(u_fringe), ), dist,)])
        i_dist = th.cat([i_dist, th.full((len(i_fringe), ), dist,)])

    nodes = th.cat([u_nodes, i_nodes])

    # 2. node labeling
    # this labeling is based on hop from starting nodes
    u_node_labels = th.stack([x*2 for x in u_dist])
    v_node_labels = th.stack([x*2+1 for x in i_dist])
    node_labels = th.cat([u_node_labels, v_node_labels])

    return nodes, node_labels


def subgraph_extraction_labeling(u_node_idx, i_node_idx, graph,
                                 hop=1, sample_ratio=1.0, max_nodes_per_hop=200,):

    # extract the h-hop enclosing subgraph nodes around link 'ind'
    nodes, node_labels = get_neighbor_nodes_labels(u_node_idx=u_node_idx, i_node_idx=i_node_idx, graph=graph, 
                                                  hop=hop, sample_ratio=sample_ratio, max_nodes_per_hop=max_nodes_per_hop)

    subgraph = dgl.node_subgraph(graph, nodes, store_ids=True) 

    subgraph.ndata['nlabel'] = one_hot(node_labels, (hop+1)*2)
    subgraph.ndata['x'] = subgraph.ndata['nlabel']

    # set edge mask to zero as to remove links between target nodes in training process
    subgraph.edata['edge_mask'] = th.ones(subgraph.number_of_edges(), dtype=th.float32)
    su = subgraph.nodes()[subgraph.ndata[dgl.NID]==u_node_idx]
    si = subgraph.nodes()[subgraph.ndata[dgl.NID]==i_node_idx]
    _, _, target_edges = subgraph.edge_ids([su, si], [si, su], return_uv=True)
    subgraph.remove_edges(target_edges)
    
    # mask target edge label
    # subgraph.edata['edge_mask'][target_edges.to(th.long)] = 0
    # subgraph.edata['label'][target_edges.to(th.long)] = 0.0
    subgraph = dgl.add_self_loop(subgraph)
    return subgraph    



#######################
# Subgraph Dataset 
#######################

class UserItemDataset(th.utils.data.Dataset):
    def __init__(self, user_item_graph: UserItemGraph,
                 keyword_edge_k=5, keyword_edge_cooc_matrix=None,
                 additional_feature=None,
                 hop=1, sample_ratio=1.0, max_nodes_per_hop=100):

        self.g_labels = user_item_graph.labels
        self.graph = user_item_graph.graph
        self.pairs = user_item_graph.user_item_pairs
        self.pairs_set = set([ (p[0].item(), p[1].item()) for p in self.pairs])

        self.min_count = keyword_edge_k
        self.keyword_edge_cooc_matrix = keyword_edge_cooc_matrix
        self.additional_feature = additional_feature

        self.hop = hop
        self.sample_ratio = sample_ratio
        self.max_nodes_per_hop = max_nodes_per_hop

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        u_idx, i_idx = self.pairs[idx]
        subgraph = subgraph_extraction_labeling(u_idx, i_idx, self.graph)

        if 'feature' in subgraph.edata.keys():
            masked_feat = th.mul(subgraph.edata['feature'], th.unsqueeze(subgraph.edata['edge_mask'],1))
            subgraph.edata['feature']= masked_feat
        
        g_label = self.g_labels[idx]
        if self.keyword_edge_cooc_matrix is not None:
            subgraph = self._add_keyword_normalized_edge(subgraph)
            # subgraph = self._add_keyword_cosin_sim_edge(subgraph)

        if self.additional_feature is not None:
            try:
                u_vector = self.additional_feature[u_idx]
                i_vector = self.additional_feature[i_idx]
            except:
                u_vector = np.zeros((768,), dtype=np.float32)
                i_vector = np.zeros((768,), dtype=np.float32)

            feature_vector = np.concatenate([u_vector, i_vector], axis=0)
            n = subgraph.number_of_nodes()
            return subgraph, feature_vector, g_label
        
        return subgraph, g_label

    def _get_etype(self, i_ntype, j_ntype, ntypes):
        if (i_ntype[0] == 1 and j_ntype[1] == 1) or (j_ntype[0] == 1 and i_ntype[1] == 1):
            return 0
        elif (i_ntype[2] == 1 and j_ntype[3] == 1) or (j_ntype[2] == 1 and i_ntype[3] == 1):
            return 6
        else:
            return 7

    def _add_keyword_normalized_edge(self, subg, max_count=100):
        oid_nid_dict = {}
        for new_id, original_id in zip(subg.nodes().tolist(), subg.ndata['_ID'].tolist()):
            oid_nid_dict[original_id] = new_id

        nids = subg.ndata['node_id'].tolist()
        ntypes = subg.ndata['x']

        pairs = list(combinations(nids, 2))
        additional_edges_que = PriorityQueue()
        for i, j in pairs:
            if i>=len(self.keyword_edge_cooc_matrix) or j>=len(self.keyword_edge_cooc_matrix):
                continue
            k_count = self.keyword_edge_cooc_matrix[i,j]
            if k_count > self.min_count:
                if k_count > max_count:
                    k_count = max_count
                additional_edges_que.put((-k_count, (i,j)))

        if additional_edges_que.empty() == True:
            return subg
        
        src, dst, etypes, keyword_cooc_counts = [], [], [], []
        n = subg.number_of_edges()//4
        for k in range(additional_edges_que.qsize()):
            if k > n :
                break
            neg_count, (i, j) = additional_edges_que.get()
            cooc_count = -neg_count
            i_ntype, j_ntype = ntypes[oid_nid_dict[i]], ntypes[oid_nid_dict[j]]
            e = self._get_etype(i_ntype, j_ntype, ntypes)
            e_vec = [0]*8
            e_vec[e] = 1 
            etypes += [e_vec, e_vec]
            keyword_cooc_counts += [cooc_count, cooc_count]
            src += [oid_nid_dict[i], oid_nid_dict[j]]
            dst += [oid_nid_dict[j], oid_nid_dict[i]]

        norm_keyword_cooc_counts = (np.array(keyword_cooc_counts)-self.min_count)/max_count
        norm_keyword_cooc_counts = np.tile(norm_keyword_cooc_counts, (8,1)).T
        n_edges = len(keyword_cooc_counts)
        edata={
            'etype_vect': th.tensor(np.array(etypes)*norm_keyword_cooc_counts, dtype=th.float32),
            'label': th.tensor(np.array([1.]*n_edges), dtype=th.float32),
            # 'edge_mask': th.tensor(norm_keyword_cooc_counts, dtype=th.float32),
        }
        subg.add_edges(src, dst, data=edata)
        return subg
    
    def _add_keyword_cosin_sim_edge(self, subg,):
        oid_nid_dict = {}
        for new_id, original_id in zip(subg.nodes().tolist(), subg.ndata['_ID'].tolist()):
            oid_nid_dict[original_id] = new_id

        nids = subg.ndata['node_id'].tolist()
        ntypes = subg.ndata['x']

        pairs = list(combinations(nids, 2))
        additional_edges_que = PriorityQueue()
        for i, j in pairs:
            if i>=len(self.keyword_edge_cooc_matrix) or j>=len(self.keyword_edge_cooc_matrix):
                continue
            cos_sim = self.keyword_edge_cooc_matrix[i,j]
            if cos_sim > 0.5:
                additional_edges_que.put((-cos_sim, (i,j)))

        src, dst, etypes, keyword_cos_sim = [], [], [], []
        n = subg.number_of_edges()//5
        if additional_edges_que.empty() == True:
            return subg

        for k in range(additional_edges_que.qsize()):
            if k > n :
                break
            neg_cos_sim, (i, j) = additional_edges_que.get()
            cos_sim = -neg_cos_sim
            # i_ntype, j_ntype = ntypes[oid_nid_dict[i]], ntypes[oid_nid_dict[j]]
            # e = self._get_etype(i_ntype, j_ntype, ntypes)
            # etypes += [e, e]
            keyword_cos_sim += [cos_sim, cos_sim]
            src += [oid_nid_dict[i], oid_nid_dict[j]]
            dst += [oid_nid_dict[j], oid_nid_dict[i]]

        n_edges = len(keyword_cos_sim)
        # print(n_edges)
        edata={
            # 'etype': th.tensor(np.array(etypes), dtype=th.int32),
            'etype': th.tensor(np.array([0]*n_edges), dtype=th.int32),
            'label': th.tensor(np.array([1.]*n_edges), dtype=th.float32),
            'edge_mask': th.tensor(keyword_cos_sim, dtype=th.float32),
        }
        subg.add_edges(src, dst, data=edata)
        return subg

    def _add_keyword_count_edge(self, subg):
        oid_nid_dict = {}
        for new_id, original_id in zip(subg.nodes().tolist(), subg.ndata['_ID'].tolist()):
            oid_nid_dict[original_id] = new_id

        nids = subg.ndata['node_id'].tolist()
        edata={
            'etype':th.tensor([0], dtype=th.int32),
            'edge_mask':th.tensor([1], dtype=th.int32),
            'label':th.tensor([1.]),
        }
        pairs = list(combinations(nids, 2))
        for i, j in pairs:
            if i>=len(self.keyword_edge_cooc_matrix) or j>=len(self.keyword_edge_cooc_matrix):
                continue
            if self.keyword_edge_cooc_matrix[i,j] > self.min_count:
                subg.add_edges(oid_nid_dict[i], oid_nid_dict[j], data=edata)
                subg.add_edges(oid_nid_dict[j], oid_nid_dict[i], data=edata)
                
        return subg

    # def _generate_keyword_graph(self, subg, min_count=5, max_count=100):
    #     oid_nid_dict = {}
    #     for new_id, original_id in zip(subg.nodes().tolist(), subg.ndata['_ID'].tolist()):
    #         oid_nid_dict[original_id] = new_id

    #     nids = subg.ndata['node_id'].tolist()
    #     pairs = list(combinations(nids, 2))
    #     additional_edges = []
    #     keyword_cooc_counts = []
    #     for i, j in pairs:
    #         if i>=len(self.keyword_edge_cooc_matrix) or j>=len(self.keyword_edge_cooc_matrix):
    #             continue
    #         k_count = self.keyword_edge_cooc_matrix[i,j]
    #         if k_count > min_count:
    #             additional_edges.append((i,j))
    #             if k_count > max_count:
    #                 k_count = max_count
    #             keyword_cooc_counts.append(k_count)
        
    #     if len(keyword_cooc_counts) == 0:
    #         keyword_subg = dgl.graph(([], []), num_nodes=len(nids))
    #         for k, v in subg.ndata.items():
    #             keyword_subg.ndata[k] = v 
    #         keyword_subg.edata['keywords'] = th.tensor([], dtype=th.float32)
    #         return dgl.add_self_loop(keyword_subg)

    #     norm_keyword_cooc_counts = 1 - np.array(keyword_cooc_counts)/max(keyword_cooc_counts)
    #     src, dst = [], [] 
    #     for i,j in additional_edges:
    #         src += [oid_nid_dict[i], oid_nid_dict[j]]
    #         dst += [oid_nid_dict[j], oid_nid_dict[i]]

    #     keyword_subg = dgl.graph((src, dst), num_nodes=len(nids))
    #     for k, v in subg.ndata.items():
    #         keyword_subg.ndata[k] = v 
    #     keyword_subg.edata['keywords'] = th.tensor([e for x in zip(*[norm_keyword_cooc_counts]*2) for e in x], dtype=th.float32)
    #     return dgl.add_self_loop(keyword_subg)



def collate_data(data):
    g_list, label_list = map(list, zip(*data))
    g = dgl.batch(g_list)
    g_label = th.stack(label_list)
    return g, g_label

def kgraph_collate_data(data):
    g_list, kg_list, label_list = map(list, zip(*data))
    g = dgl.batch(g_list)
    kg = dgl.batch(kg_list)
    g_label = th.stack(label_list)
    return g, kg, g_label

def bert_collate_data(data):
    g_list, vector_list, label_list = map(list, zip(*data))
    g = dgl.batch(g_list)
    vectors = th.stack([th.tensor(v) for v in vector_list])
    g_label = th.stack(label_list)
    return g, vectors, g_label


def get_graphs(data_path, item_kw_df=None, user_kw_df=None):
    train_df = pd.read_csv(f'{data_path}_train.csv')
    valid_df = pd.read_csv(f'{data_path}_valid.csv')
    test_df = pd.read_csv(f'{data_path}_test.csv')

    #accumulate
    valid_df = pd.concat([train_df, valid_df])
    test_df = pd.concat([valid_df, test_df])

    train_graph = UserItemGraph(label_col='rating',
                                user_col='user_id',
                                item_col='item_id',
                                text_col='text_clean',
                                df=train_df,
                                edge_idx_range=(0, len(train_df)))

    valid_graph = UserItemGraph(label_col='rating',
                                user_col='user_id',
                                item_col='item_id',
                                text_col='text_clean',
                                df=valid_df,
                                edge_idx_range=(len(train_df), len(valid_df)))

    test_graph = UserItemGraph(label_col='rating',
                               user_col='user_id',
                               item_col='item_id',
                               text_col='text_clean',
                               df=test_df,
                               edge_idx_range=(len(valid_df), len(test_df)))

    return train_graph, valid_graph, test_graph


def get_dataloader(graph, keyword_edge_cooc_matrix, keyword_edge_k=12, additional_feature=None, batch_size=32, num_workers=8 ,shuffle=True):

    graph_dataset = UserItemDataset(user_item_graph=graph, 
                                    keyword_edge_cooc_matrix=keyword_edge_cooc_matrix,
                                    keyword_edge_k=keyword_edge_k, 
                                    additional_feature=additional_feature,
                                    hop=1, sample_ratio=1.0, max_nodes_per_hop=100)

    graph_loader = th.utils.data.DataLoader(graph_dataset, batch_size=batch_size, shuffle=shuffle, 
                                            num_workers=num_workers, collate_fn=collate_data, pin_memory=True)

    if additional_feature is not None:
        graph_loader = th.utils.data.DataLoader(graph_dataset, batch_size=batch_size, shuffle=shuffle, 
                                                num_workers=num_workers, collate_fn=bert_collate_data, pin_memory=True)


    return graph_loader


if __name__=='__main__':

    data_name = 'movie'
    data_path=f'data/{data_name}/{data_name}'
    train_graph, valid_graph, test_graph = get_graphs(data_path=data_path)

    # train_loader = get_dataloader(train_graph)