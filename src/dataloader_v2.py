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
from test_data import UserItemGraphTest

def one_hot(idx, length):
    x = th.zeros([len(idx), length], dtype=th.int32)
    x[th.arange(len(idx)), idx] = 1.0
    return x

#######################
# Subgraph Extraction 
#######################
def additional_edges_from_target_0(edges, u_node, i_node):
    filtered_edges = (edges.data['original_src_idx']!=u_node).squeeze(0)
    return filtered_edges
def additional_edges_from_target_1(edges, u_node, i_node):
    filtered_edges = (edges.data['original_src_idx']!=i_node).squeeze(0)
    return filtered_edges
def additional_edges_from_target_2(edges, u_node, i_node):
    filtered_edges = (edges.data['original_dst_idx']!=u_node).squeeze(0)
    return filtered_edges
def additional_edges_from_target_3(edges, u_node, i_node):
    filtered_edges = (edges.data['original_dst_idx']!=i_node).squeeze(0)
    return filtered_edges

def get_subgraph_label(graph:dgl.graph,
                       u_node_idx:th.tensor, i_node_idx:th.tensor,
                       u_neighbors:th.tensor, i_neighbors:th.tensor,
                       sample_ratio=1.0,)->dgl.graph:
    
    

    nodes = th.cat([u_node_idx, i_node_idx, u_neighbors, i_neighbors], dim=0,) 
    nodes = nodes.type(th.int32)
    subgraph = dgl.node_subgraph(graph, nodes, store_ids=True) 
    
    remove_edge_ids_0 = subgraph.filter_edges(
        lambda edges: additional_edges_from_target_0(edges, u_node_idx, i_node_idx))
    remove_edge_ids_1 = subgraph.filter_edges(
        lambda edges: additional_edges_from_target_1(edges, u_node_idx, i_node_idx))
    remove_edge_ids_2 = subgraph.filter_edges(
        lambda edges: additional_edges_from_target_2(edges, u_node_idx, i_node_idx))
    remove_edge_ids_3 = subgraph.filter_edges(
        lambda edges: additional_edges_from_target_3(edges, u_node_idx, i_node_idx))

    a = np.intersect1d(remove_edge_ids_0, remove_edge_ids_1)
    b = np.intersect1d(remove_edge_ids_2, remove_edge_ids_3)
    c = np.intersect1d(a, b)
    remove_edge_ids = th.tensor(c, dtype=th.int32)

    subgraph.remove_edges(remove_edge_ids)
    exists = subgraph.has_edges_between(0, 1)
    exists_2 = subgraph.has_edges_between(1, 0)
    if exists == False:
        # Define new edges to add to the subgraph
        new_src_nodes = th.tensor([0], dtype=th.int32)  # Source node(s)
        new_dst_nodes = th.tensor([1], dtype=th.int32)   # Destination node(s)

        # Add new edges to the subgraph
        subgraph.add_edges(new_src_nodes, new_dst_nodes)

        # Optional: Adding edge features
        # edge_features = {'etype': th.tensor([5],dtype=th.int8)}  # Example edge feature
        subgraph.add_edges(new_src_nodes, new_dst_nodes)

    if exists_2 == False:
        # Define new edges to add to the subgraph
        new_src_nodes = th.tensor([1], dtype=th.int32)  # Source node(s)
        new_dst_nodes = th.tensor([0], dtype=th.int32)   # Destination node(s)

        # Add new edges to the subgraph
        subgraph.add_edges(new_src_nodes, new_dst_nodes)
    
    node_labels = [0,1] + [2]*len(u_neighbors) + [3]*len(i_neighbors)
    subgraph.ndata['nlabel'] = one_hot(node_labels, 4)
    subgraph.ndata['x'] = subgraph.ndata['nlabel']

    # set edge mask to zero as to remove links between target nodes in training process
    subgraph.edata['edge_mask'] = th.ones(subgraph.number_of_edges(), dtype=th.float32)
    subgraph.edata['is_target_edge'] = th.zeros(subgraph.number_of_edges(), dtype=th.float32)
    
    target_edges = subgraph.edge_ids([0, 1], [1, 0], return_uv=False)
    subgraph.edata['is_target_edge'][target_edges] = 1.0
    subgraph= dgl.remove_edges(subgraph,target_edges)
    # subgraph = dgl.add_self_loop(subgraph)
    return subgraph    



#######################
# Subgraph Dataset 
#######################

class UserItemDataset(th.utils.data.Dataset):
    def __init__(self, user_item_graph: UserItemGraph,
                 keyword_edge_min_cooc=5, 
                 keyword_edge_cooc_matrix=None,
                 sample_ratio=1.0, max_nodes_per_hop=100):

        self.g_labels = user_item_graph.labels
        self.graph = user_item_graph.graph
        self.pairs = user_item_graph.user_item_pairs
        self.nid_neghibor_dict = user_item_graph.nid_neghibor_dict
        self.pairs_set = set([ (p[0].item(), p[1].item()) for p in self.pairs])

        self.keyword_edge_min_cooc = keyword_edge_min_cooc
        self.keyword_edge_cooc_matrix = keyword_edge_cooc_matrix

        self.sample_ratio = sample_ratio
        self.max_nodes_per_hop = max_nodes_per_hop

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        u_idx, i_idx = self.pairs[idx]
        u_neighbors, i_neighbors = self.nid_neghibor_dict[u_idx.item()], self.nid_neghibor_dict[i_idx.item()]
        u_neighbors, i_neighbors = u_neighbors[-self.max_nodes_per_hop:], i_neighbors[-self.max_nodes_per_hop:]
        u_neighbors = u_neighbors[u_neighbors!=i_idx.item()]
        i_neighbors = i_neighbors[i_neighbors!=u_idx.item()]
        subgraph = get_subgraph_label(graph = self.graph,
                                      u_node_idx=u_idx.unsqueeze(0), 
                                      i_node_idx=i_idx.unsqueeze(0), 
                                      u_neighbors=u_neighbors, 
                                      i_neighbors=i_neighbors,           
                                      sample_ratio=self.sample_ratio,
                                    )

        if 'feature' in subgraph.edata.keys():
            masked_feat = th.mul(subgraph.edata['feature'], th.unsqueeze(subgraph.edata['edge_mask'],1))
            subgraph.edata['feature']= masked_feat
        
        g_label = self.g_labels[idx]
        if self.keyword_edge_cooc_matrix is not None:
            subgraph = self._add_keyword_normalized_edge(subgraph)
            # subgraph = self._add_keyword_cosin_sim_edge(subgraph)
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
        if type(self.keyword_edge_cooc_matrix) is not dict:
            for i, j in pairs:
                if i>=len(self.keyword_edge_cooc_matrix) or j>=len(self.keyword_edge_cooc_matrix):
                    continue
                k_count = self.keyword_edge_cooc_matrix[i,j]
                if k_count > self.keyword_edge_min_cooc:
                    if k_count > max_count:
                        k_count = max_count
                    additional_edges_que.put((-k_count, (i,j)))
        else:
            for i, j in pairs:
                key = str(i)+'_'+str(j)
                if key in self.keyword_edge_cooc_matrix:
                    k_count = self.keyword_edge_cooc_matrix.get(key)
                    if k_count > self.keyword_edge_min_cooc:
                        if k_count > max_count:
                            k_count = max_count
                        additional_edges_que.put((-k_count, (i,j)))
                else:
                    continue

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

        norm_keyword_cooc_counts = (np.array(keyword_cooc_counts)-self.keyword_edge_min_cooc)/max_count
        norm_keyword_cooc_counts = np.tile(norm_keyword_cooc_counts, (8,1)).T
        n_edges = len(keyword_cooc_counts)
        edata={
            'etype': th.tensor(np.array(etypes)*norm_keyword_cooc_counts, dtype=th.float32),
            'w': th.tensor(np.array([1.]*n_edges), dtype=th.float32),
            'label': th.tensor(np.array([1.]*n_edges), dtype=th.float32),
        }
        subg.add_edges(src, dst, data=edata)
        return subg



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


def get_graphs(data_path, item_cooc_df=None, user_cooc_df=None, user_item_cooc_df=None):
    train_df = pd.read_csv(f'{data_path}_train_new.csv')
    valid_df = pd.read_csv(f'{data_path}_valid_new.csv')
    test_df = pd.read_csv(f'{data_path}_test_new.csv')

    valid_df = pd.concat([train_df, valid_df])
    test_df = pd.concat([train_df, test_df])
    
    train_graph = UserItemGraph(label_col='rating',
                                user_col='user_id',
                                item_col='item_id',
                                df=train_df,
                                user_cooc_edge_df=user_cooc_df, item_cooc_edge_df=item_cooc_df, 
                                user_item_cooc_edge_df=user_item_cooc_df,
                                edge_idx_range=(0, len(train_df)))

    valid_graph = UserItemGraph(label_col='rating',
                                user_col='user_id',
                                item_col='item_id',
                                df=valid_df,
                                user_cooc_edge_df=user_cooc_df, item_cooc_edge_df=item_cooc_df, 
                                user_item_cooc_edge_df=user_item_cooc_df,
                                edge_idx_range=(len(train_df), len(valid_df)))

    test_graph = UserItemGraph(label_col='rating',
                               user_col='user_id',
                               item_col='item_id',
                               df=test_df,
                               user_cooc_edge_df=user_cooc_df, item_cooc_edge_df=item_cooc_df, 
                               user_item_cooc_edge_df=user_item_cooc_df,
                               edge_idx_range= (len(train_df), len(test_df)))

    return train_graph, valid_graph, test_graph


def get_dataloader(graph, keyword_edge_cooc_matrix=None, keyword_edge_k=12, additional_feature=None, batch_size=32, num_workers=8 ,shuffle=True):

    graph_dataset = UserItemDataset(user_item_graph=graph, 
                                    keyword_edge_cooc_matrix=keyword_edge_cooc_matrix,
                                    keyword_edge_min_cooc=keyword_edge_k, 
                                    sample_ratio=1.0, max_nodes_per_hop=200)

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
    train_loader = get_dataloader(train_graph,)
    i = 0
    for g, l in train_loader:
        i+=1
        print(g, l)
        if i > 100:
            break