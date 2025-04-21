"""build graph with edge features"""
from collections import defaultdict
import numpy as np
import pandas as pd
import torch as th

import dgl 

import pandas as pd
import numpy as np
import scipy.sparse as sp

def one_hot(idx, length):
    x = [0]*length
    x[idx] = 1
    return x

#######################
# Build graph
#######################
class UserItemGraph():
    """
    Build user-item graph for training 
    Bulid Homogeneous graph with df
    only user user-item pairs in range when extracting subgraph
    """
    def __init__(self, label_col:str, user_col:str, item_col:str,
                 df:pd.DataFrame, edge_idx_range:tuple,
                 user_cooc_edge_df=None, item_cooc_edge_df=None, user_item_cooc_edge_df=None
                 ):
        df = df.copy()
        # df['etype'] = df[label_col]

        self._num_user = max(df[user_col].unique()) + 1
        # print("num_users: ", self._num_user)
        self._num_item = max(df[item_col].unique()) + 1
        # print("num_items: ", self._num_item)
        self._num_label = len(df[label_col].unique())
        
        df[item_col] += self._num_user
        u_idx, i_idx, = df[user_col].to_numpy(), df[item_col].to_numpy(), 
        etypes = df[label_col].to_numpy()

        weights = np.ones_like(etypes)

        labels = (df[label_col].to_numpy() - 1)/4

        # use whole data to build main graph
        # add bidirect edges
        num_nodes = self._num_user + self._num_item
        src_nodes = np.concatenate((u_idx, i_idx))
        dst_nodes = np.concatenate((i_idx, u_idx))
        labels = np.concatenate((labels, labels))
        etypes = np.concatenate((etypes, etypes))
        weights = np.concatenate((weights, weights))

        # print('edges ', len(df))
        # print('nodes ', num_nodes)
        # print('max id ', max(src_nodes), max(dst_nodes) )

        sp_mat = sp.coo_matrix((labels,(src_nodes, dst_nodes)), shape=(num_nodes, num_nodes))
        self.graph =dgl.from_scipy(sp_mat=sp_mat, idtype=th.int32)

        self.graph.ndata['node_id'] = th.tensor(list(range(num_nodes)), dtype=th.int32)

        self.graph.edata['original_src_idx'] = th.tensor(src_nodes, dtype=th.int32)
        self.graph.edata['original_dst_idx'] = th.tensor(dst_nodes, dtype=th.int32)
        self.graph.edata['label'] = th.tensor(labels, dtype=th.float32)
        self.graph.edata['etype'] = th.tensor(etypes, dtype=th.int8)
        self.graph.edata['w'] = th.tensor(weights, dtype=th.float32)

        #extract subgraph pair idx
        start, end = edge_idx_range

        self.user_indices = th.tensor(u_idx[start:end], dtype=th.int32)
        self.item_indices = th.tensor(i_idx[start:end], dtype=th.int32)
        self.labels = th.tensor(labels[start:end], dtype=th.float32)

        self.user_item_pairs = self.get_user_item_pairs()
        nid_neghibor_dict = defaultdict(list)
        for u, i in zip(u_idx, i_idx):
            nid_neghibor_dict[i].append(u)
            nid_neghibor_dict[u].append(i)
        
        self.nid_neghibor_dict = dict()
        for k, v in nid_neghibor_dict.items():
            self.nid_neghibor_dict[k] = th.tensor(v)

        # print('original edges : ', self.graph.number_of_edges())
        if user_cooc_edge_df is not None:
            print('add user cooc edge')
            self._add_additional_edge(user_cooc_edge_df, 6)
            print('add user edges : ', self.graph.number_of_edges())

        if item_cooc_edge_df is not None:
            print('add item cooc edge')
            item_cooc_edge_df['u'] += self._num_user
            item_cooc_edge_df['v'] += self._num_user
            self._add_additional_edge(item_cooc_edge_df, 7)         
            print('add item edges : ', self.graph.number_of_edges())

        if user_item_cooc_edge_df is not None:
            print('add user-item cooc edge')
            user_item_cooc_edge_df['i'] += self._num_user
            self._add_additional_edge(user_item_cooc_edge_df, 8)
            print('add user-item edges : ', self.graph.number_of_edges())
            
            
    def _add_additional_edge(self, cooc_edge_df, etype):
        # n = subg.number_of_edges()//4
        cols = list(cooc_edge_df.columns)
        u_col, v_col = cols[-2], cols[-1] 
        # print(u_col, v_col)
        n = len(cooc_edge_df)
        us, vs = cooc_edge_df[u_col].to_list(), cooc_edge_df[v_col].to_list()
        src, dst, etypes = us+vs, vs+us, [etype]*n*2

        edata={
            'original_src_idx': th.tensor(np.array(src), dtype=th.int32),
            'original_dst_idx': th.tensor(np.array(dst), dtype=th.int32),
            'etype': th.tensor(np.array(etypes), dtype=th.int8),
            'w': th.tensor(np.array([1.]*n*2), dtype=th.float32),
            'label': th.tensor(np.array([1.]*n*2), dtype=th.float32),
        }

        self.graph.add_edges(src, dst, data=edata)

    def get_user_item_pairs(self):
        pairs = []
        for u, i in zip(self.user_indices, self.item_indices):
            pairs.append((u,i))
        return pairs
