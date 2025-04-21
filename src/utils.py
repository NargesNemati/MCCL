


import logging
from dataloader_v2 import get_subgraph_label, collate_data, bert_collate_data
from user_item_graph import UserItemGraph
import pandas as pd
import torch.nn.functional as F
import math

def get_logger(name, path):

    logger = logging.getLogger(name)
    
    if len(logger.handlers) > 0:
        return logger # Logger already exists

    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s - %(message)s")

    console = logging.StreamHandler()
    file_handler = logging.FileHandler(filename=path)
    
    console.setLevel(logging.DEBUG)
    file_handler.setLevel(logging.INFO)

    console.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console)
    logger.addHandler(file_handler)

    return logger



from easydict import EasyDict
import yaml


def get_args_from_yaml(yaml_path):

    with open('../train_configs/common_configs.yaml') as f:
        common_cfgs = yaml.load(f, Loader=yaml.FullLoader)
    data_cfg = common_cfgs['dataset']
    model_cfg = common_cfgs['model']
    train_cfg = common_cfgs['train']
    print(yaml_path)

    with open(yaml_path) as f:
        cfgs = yaml.load(f, Loader=yaml.FullLoader)
    exp_data_cfg = cfgs.get('dataset', dict())
    exp_model_cfg = cfgs.get('model', dict())
    exp_train_cfg = cfgs.get('train', dict())

    for k, v in exp_data_cfg.items():
        data_cfg[k] = v
    for k, v in exp_model_cfg.items():
        model_cfg[k] = v
    for k, v in exp_train_cfg.items():
        train_cfg[k] = v
    args = EasyDict(
        {   
            'key': cfgs['key'],

            'dataset': data_cfg.get('name'),
            'datasets': data_cfg.get('datasets'),
            'keywords': data_cfg.get('keywords'),
            'item_cooc_edge_df': data_cfg.get('item_cooc_edge_df'),
            'user_cooc_edge_df': data_cfg.get('user_cooc_edge_df'),
            'user_item_cooc_edge_df': data_cfg.get('user_item_cooc_edge_df'),
            
            # model configs
            'model_type': [ model for model in model_cfg.get('type') ],
            'hop': model_cfg['hop'],
            'in_nfeats': model_cfg.get('in_nfeats'),
            'out_nfeats': model_cfg.get('out_nfeats'),
            'in_efeats': model_cfg.get('in_efeats'),
            'out_efeats': model_cfg.get('out_efeats'),
            'num_heads': model_cfg.get('num_heads'),
            'node_features': model_cfg.get('node_features'),
            'parameters': model_cfg.get('parameters'),
            'num_relations': model_cfg.get('num_relations', 8),
            'edge_dropout': model_cfg['edge_dropout'],

            'latent_dims': model_cfg.get('latent_dims'), # baseline model

            #train configs
            'device':train_cfg['device'],
            'log_dir': train_cfg['log_dir'],
            'log_interval': train_cfg.get('log_interval'),
            'train_lrs': [ float(lr) for lr in train_cfg.get('learning_rates') ],
            'train_epochs': train_cfg.get('epochs'),
            'batch_size': train_cfg['batch_size'],
            'weight_decay': train_cfg.get('weight_decay', 0),
            'lr_decay_step': train_cfg.get('lr_decay_step'),
            'lr_decay_factor': train_cfg.get('lr_decay_factor'),
            'ndcg_k': train_cfg.get('ndcg_k'),

        }
    )

    return args

import numpy as np
import torch as th
import random
import numpy as np
from collections import defaultdict





class UserItemDataset(th.utils.data.Dataset):
    def __init__(self, user_item_graph: UserItemGraph,
                 users, unique_items,
                 flag,
                 original_pair_labales,
                 user_item_train_dict = None,
                 keyword_edge_min_cooc=5, 
                 keyword_edge_cooc_matrix=None,
                 sample_ratio=1.0, max_nodes_per_hop=100):

 
        users = users
        unique_items = unique_items
        # self.g_labels = user_item_graph.labels
        num_users = user_item_graph._num_user 
        num_items = user_item_graph._num_item 

        # self.src_target_nodes = user_item_graph.user_indices
        # self.dst_target_nodes = user_item_graph.item_indices
        self.graph = user_item_graph.graph
        self.pairs = []
        # for item in unique_items:
        #     # self.pairs.append((th.tensor(item+num_users, dtype=th.int32), th.tensor(user, dtype=th.int32)))
        #     self.pairs.append((th.tensor(user, dtype=th.int32), th.tensor(item+num_users, dtype=th.int32)))
        # user_tensor = th.tensor(user, dtype=th.int32)

    # Create pairs using list comprehension
        if flag == 0:

            for i in range(min(len(users), 2000)):
                user_tensor = th.tensor(users[i], dtype=th.int32)
                self.pairs.extend(
                    (user_tensor, th.tensor(item + num_users, dtype=th.int32)) 
                    for item in unique_items[i]
                )

        if flag == 1:
            for i in range(len(users)):
                user_tensor = th.tensor(users[i], dtype=th.int32)
                self.pairs.extend(
                    (user_tensor, th.tensor(item + num_users, dtype=th.int32)) 
                    for item in unique_items[i]
                )
        
        second_pairs = [ (p[0].item(), p[1].item()) for p in self.pairs]

        # self.g_labels = []
        # for i in range(len(second_pairs)):
        #     if second_pairs[i] in original_pair_labales:
        #         self.g_labels.append(original_pair_labales[second_pairs[i]][0])
        #     else:
        #         self.g_labels.append(th.tensor(0))
        
        original_keys = set(original_pair_labales.keys())  # Create a set of keys for fast lookup

        g_labels = [
            original_pair_labales[sp][0] if sp in original_keys else th.tensor(0)
            for sp in second_pairs
        ]

        # self.g_label = th.tensor(self.g_labels)
        self.g_labels = th.stack(g_labels)
        
        # self.pairs = user_item_graph.user_item_pairs
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
            'etype_vect': th.tensor(np.array(etypes)*norm_keyword_cooc_counts, dtype=th.float32),
            'label': th.tensor(np.array([1.]*n_edges), dtype=th.float32),
        }
        subg.add_edges(src, dst, data=edata)
        return subg


from torch.utils.data import ConcatDataset




def test_ndcg_dataset(args):
    dataset = {}
    data_path = f'../data/{args.data_name}/{args.data_name}'
    train_df = pd.read_csv(f'{data_path}_train_new.csv')
    valid_df = pd.read_csv(f'{data_path}_valid_new.csv')
    test_df = pd.read_csv(f'{data_path}_valid_new.csv')
    user_item_train_dict = defaultdict(list)
    for user_id, item_id in zip(train_df['user_id'], train_df['item_id']):
        user_item_train_dict[user_id].append(item_id)


    #accumulate
    valid_df = pd.concat([train_df, valid_df])
    test_df = pd.concat([train_df, test_df])
    
    # test_df_for_pairs = pd.read_csv(f'{data_path}_test.csv')
    test_df_for_pairs = pd.read_csv(f'{data_path}_test_new.csv')
    unique_users = test_df_for_pairs['user_id'].unique()
    unique_items = set(train_df['item_id'].unique())
    # Ensure the DataFrame is sorted by user_id and ts
    test_df_for_pairs = test_df_for_pairs.sort_values(by=["user_id", "ts"])

    # Get the last interaction for each user
    last_interactions = test_df_for_pairs.groupby("user_id").last().reset_index()
    last_interacted_items = set(last_interactions["item_id"])
    # unique_items = set(last_interactions['item_id'].unique())
    non_last_interacted_items = list(unique_items - last_interacted_items)
    random_items = np.random.choice(non_last_interacted_items, size=99, replace=False)
    new_test_df = pd.concat([test_df, last_interactions])


    # unique_userstrain = train_df['user_id'].unique()

    # unique_items = test_df['item_id'].unique()

    
    data_path = f'../data/{args.data_name}/{args.data_name}'
    item_cooc_edge_df, user_cooc_edge_df, user_item_cooc_edge_df = None, None, None 
    if args.item_cooc_edge_df != 'None'  :
        item_cooc_edge_df = pd.read_csv(f'../data/{args.data_name}/{args.item_cooc_edge_df}_cooc.csv', index_col=0) 
    if args.user_cooc_edge_df != 'None'  :
        user_cooc_edge_df = pd.read_csv(f'../data/{args.data_name}/{args.user_cooc_edge_df}_cooc.csv', index_col=0) 
    if args.user_item_cooc_edge_df != 'None'  :
        user_item_cooc_edge_df = pd.read_csv(f'../data/{args.data_name}/{args.user_item_cooc_edge_df}_cooc.csv', index_col=0) 
      
            
    test_rank_graph = UserItemGraph(label_col='rating',
                                user_col='user_id',
                                item_col='item_id',
                                df=new_test_df,
                                user_cooc_edge_df=user_cooc_edge_df, item_cooc_edge_df=item_cooc_edge_df, 
                                user_item_cooc_edge_df=user_item_cooc_edge_df,
                                edge_idx_range=(len(train_df), len(new_test_df)))
    
    original_pairs = test_rank_graph.user_item_pairs
    labels = test_rank_graph.labels
    lab_pair = {}
    original_pairs = [ (p[0].item(), p[1].item()) for p in original_pairs]
    for i in range(len(original_pairs)):
        lab_pair[original_pairs[i]] = [labels[i]]


    counter = 0
    item_list = []
    item_list_ndcg = []

    for i in range(min(len(unique_users), 2000)):  # Iterate over users
        # filtered_numbers = np.array([num for num in unique_items if num not in user_item_train_dict[unique_users[i]]])

        user_id = unique_users[i]  # Assume unique_users[i] is defined

        last_item_for_user = last_interactions[last_interactions['user_id'] == user_id]['item_id'].values
        filtered_numbers = np.append(random_items, last_item_for_user[0])
        item_list.append(filtered_numbers)

        # filtered_numbers_ndcg = np.array([num for num in unique_items if num not in user_item_train_dict[unique_users[i]]])



    graph_dataset = UserItemDataset(user_item_graph=test_rank_graph,
                                    users = unique_users, unique_items = item_list,
                                    flag = 0,
                                    original_pair_labales = lab_pair,
                                    keyword_edge_cooc_matrix=None,
                                    keyword_edge_min_cooc=None, 
                                    sample_ratio=1.0, max_nodes_per_hop=200)
    
    graph_loader = th.utils.data.DataLoader(graph_dataset, batch_size=100, shuffle=False, 
                                        num_workers=8, collate_fn=collate_data, pin_memory=True)


# if additional_feature is not None:
#     graph_loader = th.utils.data.DataLoader(graph_dataset, batch_size=128, shuffle=False, 
#                                             num_workers=3, collate_fn=bert_collate_data, pin_memory=True)      
    
    # dataset[i] = graph_loader

    return graph_loader


def test_rel_ndcg_dataset(args):
    dataset = {}
    data_path = f'../data/{args.data_name}/{args.data_name}'
    train_df = pd.read_csv(f'{data_path}_train_new.csv')
    valid_df = pd.read_csv(f'{data_path}_valid_new.csv')
    test_df = pd.read_csv(f'{data_path}_valid_new.csv')
    user_item_train_dict = defaultdict(list)
    for user_id, item_id in zip(train_df['user_id'], train_df['item_id']):
        user_item_train_dict[user_id].append(item_id)

    #accumulate
    valid_df = pd.concat([train_df, valid_df])
    test_df = pd.concat([train_df, test_df])
    
    # test_df_for_pairs = pd.read_csv(f'{data_path}_test.csv')
    test_df_for_pairs = pd.read_csv(f'{data_path}_test_new.csv')
    unique_users = test_df_for_pairs['user_id'].unique()
    unique_items = set(test_df['item_id'].unique())

    
    data_path = f'../data/{args.data_name}/{args.data_name}'
    item_cooc_edge_df, user_cooc_edge_df, user_item_cooc_edge_df = None, None, None 
    if args.item_cooc_edge_df != 'None'  :
        item_cooc_edge_df = pd.read_csv(f'../data/{args.data_name}/{args.item_cooc_edge_df}_cooc.csv', index_col=0) 
    if args.user_cooc_edge_df != 'None'  :
        user_cooc_edge_df = pd.read_csv(f'../data/{args.data_name}/{args.user_cooc_edge_df}_cooc.csv', index_col=0) 
    if args.user_item_cooc_edge_df != 'None'  :
        user_item_cooc_edge_df = pd.read_csv(f'../data/{args.data_name}/{args.user_item_cooc_edge_df}_cooc.csv', index_col=0) 
      
            
    test_rank_graph = UserItemGraph(label_col='rating',
                                user_col='user_id',
                                item_col='item_id',
                                df=test_df,
                                user_cooc_edge_df=user_cooc_edge_df, item_cooc_edge_df=item_cooc_edge_df, 
                                user_item_cooc_edge_df=user_item_cooc_edge_df,
                                edge_idx_range=(len(train_df), len(test_df)))
    
    original_pairs = test_rank_graph.user_item_pairs
    labels = test_rank_graph.labels
    lab_pair = {}
    original_pairs = [ (p[0].item(), p[1].item()) for p in original_pairs]
    for i in range(len(original_pairs)):
        lab_pair[original_pairs[i]] = [labels[i]]


    counter = 0
    item_list = []
    item_list_ndcg = []
    counter = 0

    for i in range(len(unique_users)):  # Iterate over users
        # filtered_numbers = np.array([num for num in unique_items if num not in user_item_train_dict[unique_users[i]]])
        filtered_numbers = np.array([num for num in unique_items])
        item_list.append(filtered_numbers)

        # user_id = unique_users[i] 

    graph_dataset = UserItemDataset(user_item_graph=test_rank_graph,
                                    users=unique_users, unique_items = item_list,
                                    flag=1,
                                    original_pair_labales = lab_pair,
                                    user_item_train_dict = user_item_train_dict,
                                    keyword_edge_cooc_matrix=None,
                                    keyword_edge_min_cooc=None, 
                                    sample_ratio=1.0, max_nodes_per_hop=200)
    
    graph_loader = th.utils.data.DataLoader(graph_dataset, batch_size=len(item_list[0]), shuffle=False, 
                                        num_workers=8, collate_fn=collate_data, pin_memory=True)


    return graph_loader



def compute_ndcg_mrr(user_predictions, user_labels, top_k):
    # user_predictions = th.stack(user_predictions)

    sorted_preds, sorted_indices = th.sort(user_predictions, descending=True)
    

    # sorted_indices = [t.item() for t in sorted_indices]
    sorted_predicted_labels = [user_labels[i] for i in sorted_indices[:top_k]]

    dcg = 0.0
    for i in range(top_k):
        dcg += (sorted_predicted_labels[i] * 4 + 1) / np.log2(i + 2)

    sorted_ideal_labels = sorted(user_labels, reverse=True)
    ideal_ndcg = 0
    for i in range(top_k):
        ideal_ndcg += (sorted_ideal_labels[i] * 4 + 1 ) / np.log2(i + 2) 
   
    ndcg = dcg / ideal_ndcg if ideal_ndcg > 0 else 0

    # Compute MRR for the user
    mrr = 0.0
    best = sorted_ideal_labels[0]

    # random_negative = reduce_zeros(sorted_predicted_labels, target_zeros=2000)
    new_ndcg = 0.0
    for i in range(top_k):
        if sorted_predicted_labels[i] == best:  
            mrr = 1 / (i + 1)
            new_ndcg = 1 / math.log(i+2 , 2)
            break
    if type(ndcg) == int:
        ndcg = th.tensor(ndcg).to(0)

    mrr = th.tensor(mrr).to(0)
    new_ndcg = th.tensor(new_ndcg).to(0)
    return  ndcg, mrr, new_ndcg

import time
def evaluate_user_separately(model, model_1, model_2, args, datasets_dict, top_k=10):
    model.eval()
    if model_1 is not None:
        model_1.eval()
        model_2.eval()
    all_ndcgs = []
    all_mrrs = []
    all_base_ndcg = []
    counter = 0
    # for i in datasets_dict.keys():
    #     counter += 1
    # user_predictions = []
    # user_labels = []
        # src_node_ids = []
        # dst_node_ids = []

    loader = datasets_dict

    # start_time = time.time()
    # for batch in loader:
    # for iter_idx, batch in enumerate(loader, start=1):
    # counter = 0
    counter = 0
    for batch in loader:

        with th.no_grad():
            inputs = batch[0].to(args.device)
            if model_1 is not None:
                pred, _, emb1 = model_1(inputs)
                pred, _, emb2 = model_2(inputs)
                loss, target_predictions = model(emb1, emb2, inputs, tau=0.5)
            else:
                # target_predictions = target_predictions.squeeze(1)
                target_predictions, _, _ = model(inputs)
        labels = batch[1].to(args.device) * 4 + 1
        # user_predictions.extend(target_predictions)
        # user_labels.extend(labels)
        # start_time = time.time()
        user_predictions = th.tensor(target_predictions).to(args.device)
        user_predictions = target_predictions
        user_labels = th.tensor(labels).to(args.device)
        ndcg, mrr, new_ndcg = compute_ndcg_mrr(user_predictions, user_labels, top_k)

        all_ndcgs.append(ndcg)
        all_mrrs.append(mrr)
        all_base_ndcg.append(new_ndcg)

    # if counter % 200 == 0:

    #     mean_ndcgs = th.stack(all_ndcgs)
    #     mean_mrr = th.stack(all_mrrs)
    #     mean_base_ndcg = th.stack(all_base_ndcg)
    #     # all_ndcgs = th.stack(all_ndcgs)


    all_ndcgs = th.stack(all_ndcgs)
    all_mrrs = th.stack(all_mrrs)
    all_base_ndcg = th.stack(all_base_ndcg)

        # Now calculate the mean
    return th.mean(all_ndcgs), th.mean(all_mrrs), th.mean(all_base_ndcg)




import numpy as np
import torch as th

# def evaluate(model, loader, device):
#     # Set model to evaluation mode
#     model.eval()
    
#     mse = 0.  # Mean squared error accumulator
#     lenn = 0   # Total number of samples
#     all_lables = []
#     all_preds = []
#     iter_mse = 0
#     iter_cnt = 0

#     for batch in loader:
#         with th.no_grad():
#             # Assuming batch[0] is input data and batch[1] is labels
#             targets, _ = model(batch[0].to(device))  # Ignore reg_loss if not needed           
#         labels = batch[1].to(device)  # Rescale labels       
#         # Rescale predictions in the same way as labels
#         preds = targets 
    #     iter_mse += (((preds*4+1) - (labels*4+1)) ** 2).sum().item()
    #     iter_cnt += preds.shape[0]

    # # all_lables = th.cat(all_lables, 0)
    # # all_preds = th.cat(all_preds, 0)

    # # mse = ((all_preds - all_lables) ** 2).sum().item()
    # # len_test = len(all_lables)
    # return math.sqrt(iter_mse/iter_cnt)

def evaluate(model, model_1, model_2, loader, device):
    # Evaluate RMSE
    model.eval()
    if model_2 is not None:
        model_1.eval()
        model_2.eval()
    mse = 0.
    for batch in loader:
        with th.no_grad():
            inputs = batch[0].to(device)
            if model_2 is not None:
                pred, _, emb1 = model_1(inputs)
                pred, _, emb2 = model_2(inputs)
                loss, preds = model(emb1, emb2, inputs, tau=0.5)
            else:
              preds, _,_ = model(inputs)
        labels = batch[1].to(device)
        mse += (((preds *4 +1) - (labels*4 +1)) ** 2).sum().item()
    mse /= len(loader.dataset)
    return np.sqrt(mse)







def feature_evaluate(model, loader, device):
    # Evaluate RMSE
    model.eval()
    mse = 0.
    for batch in loader:
        
        with th.no_grad():
            targets, reg_loss = model(batch[0].to(device), batch[1].to(device))
            print(targets)
        labels = batch[2].to(device)

        mse += ((targets - labels) ** 2).sum().item()
    mse /= len(loader.dataset)
    return np.sqrt(mse)

import torch.nn as nn


class UserItemDataset_2(th.utils.data.Dataset):
    def __init__(self, user_item_graph: UserItemGraph,
                 user, unique_items,
                 original_pair_labales,
                 keyword_edge_min_cooc=5, 
                 keyword_edge_cooc_matrix=None,
                 sample_ratio=1.0, max_nodes_per_hop=100):

 
        self.user = user
        self.unique_items = unique_items
        # self.g_labels = user_item_graph.labels
        num_users = user_item_graph._num_user 
        num_items = user_item_graph._num_item 

        self.src_target_nodes = user_item_graph.user_indices
        self.dst_target_nodes = user_item_graph.item_indices
        self.graph = user_item_graph.graph
        self.pairs = []
        # for item in unique_items:
        #     # self.pairs.append((th.tensor(item+num_users, dtype=th.int32), th.tensor(user, dtype=th.int32)))
        #     self.pairs.append((th.tensor(user, dtype=th.int32), th.tensor(item+num_users, dtype=th.int32)))
        user_tensor = th.tensor(user, dtype=th.int32)

    # Create pairs using list comprehension
        self.pairs.extend(
            (user_tensor, th.tensor(item + num_users, dtype=th.int32)) 
            for item in unique_items
        )
        
        second_pairs = [ (p[0].item(), p[1].item()) for p in self.pairs]

        # self.g_labels = []
        # for i in range(len(second_pairs)):
        #     if second_pairs[i] in original_pair_labales:
        #         self.g_labels.append(original_pair_labales[second_pairs[i]][0])
        #     else:
        #         self.g_labels.append(th.tensor(0))
        
        original_keys = set(original_pair_labales.keys())  # Create a set of keys for fast lookup

        self.g_labels = [
            original_pair_labales[sp][0] if sp in original_keys else th.tensor(0)
            for sp in second_pairs
        ]

        # self.g_label = th.tensor(self.g_labels)
        self.g_labels = th.stack(self.g_labels)
        


        # self.pairs = user_item_graph.user_item_pairs
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
            'etype_vect': th.tensor(np.array(etypes)*norm_keyword_cooc_counts, dtype=th.float32),
            'label': th.tensor(np.array([1.]*n_edges), dtype=th.float32),
        }
        subg.add_edges(src, dst, data=edata)
        return subg


def compute_ndcg_mrr_2(user_predictions, user_labels, top_k):

    sorted_preds, sorted_indices = th.sort(user_predictions, descending=True)

    sorted_predicted_labels = [user_labels[i] for i in sorted_indices]

    dcg = 0.0

    for i in range(min(top_k, len(sorted_preds))):
        dcg += (sorted_predicted_labels[i]) / np.log2(i + 2)

    sorted_ideal_labels = sorted(user_labels, reverse=True)
    ideal_ndcg = 0

    for i in range(min(top_k, len(sorted_ideal_labels))):
        ideal_ndcg += (sorted_ideal_labels[i]) / np.log2(i + 2) 
   
    ndcg = dcg / ideal_ndcg if ideal_ndcg > 0 else 0

    # Compute MRR for the user
    mrr = 0.0
    best = sorted_ideal_labels[0]
    new_ndcg = 0.0
    for i in range(top_k):
        if sorted_predicted_labels[i] == best:  
            mrr = 1 / (i + 1)
            new_ndcg = 1 / math.log(i+2 , 2)
            break
    if type(ndcg) == int:
        ndcg = th.tensor(ndcg).to(0)

    mrr = th.tensor(mrr).to(0)
    new_ndcg = th.tensor(new_ndcg).to(0)
    return  ndcg, mrr, new_ndcg


def evaluate_user_separately_2(model, model_1, model_2, args, top_k=10):
    model.eval()
    if model_1 is not None:
        model_1.eval()
        model_2.eval()
    all_ndcgs = []
    all_mrrs = []
    all_base_ndcg = []
    data_path = f'../data/{args.data_name}/{args.data_name}'
    train_df = pd.read_csv(f'{data_path}_train_new.csv')
    valid_df = pd.read_csv(f'{data_path}_valid_new.csv')
    test_df = pd.read_csv(f'{data_path}_test_new.csv')
    user_item_train_dict = defaultdict(list)
    for user_id, item_id in zip(train_df['user_id'], train_df['item_id']):
        user_item_train_dict[user_id].append(item_id)


    #accumulate
    valid_df = pd.concat([train_df, valid_df])
    test_df = pd.concat([valid_df, test_df])
    test_df_for_pairs = pd.read_csv(f'{data_path}_test_new.csv')
    unique_users = test_df_for_pairs['user_id'].unique()
    # unique_userstrain = train_df['user_id'].unique()

    unique_items = test_df['item_id'].unique()

    
    data_path = f'../data/{args.data_name}/{args.data_name}'
    item_cooc_edge_df, user_cooc_edge_df, user_item_cooc_edge_df = None, None, None 
    if args.item_cooc_edge_df != 'None' :
        item_cooc_edge_df = pd.read_csv(f'../data/{args.data_name}/{args.item_cooc_edge_df}_cooc.csv', index_col=0) 
    if args.user_cooc_edge_df != 'None' :
        user_cooc_edge_df = pd.read_csv(f'../data/{args.data_name}/{args.user_cooc_edge_df}_cooc.csv', index_col=0) 
    if args.user_item_cooc_edge_df != 'None' :
        user_item_cooc_edge_df = pd.read_csv(f'../data/{args.data_name}/{args.user_item_cooc_edge_df}_cooc.csv', index_col=0) 
    
    # print('--------------------------')
    # print(item_cooc_edge_df)
    # print('--------------------------')   
      
            
    test_rank_graph = UserItemGraph(label_col='rating',
                                user_col='user_id',
                                item_col='item_id',
                                df=test_df,
                                user_cooc_edge_df=user_cooc_edge_df, item_cooc_edge_df=item_cooc_edge_df, 
                                user_item_cooc_edge_df=user_item_cooc_edge_df,
                                edge_idx_range=(len(valid_df), len(test_df)))
    
    original_pairs = test_rank_graph.user_item_pairs
    labels = test_rank_graph.labels
    lab_pair = {}
    original_pairs = [ (p[0].item(), p[1].item()) for p in original_pairs]
    for i in range(len(original_pairs)):
        lab_pair[original_pairs[i]] = [labels[i]]


    counter = 0
    item_list = []

    # for i in range(len(unique_users)):  # Iterate over users
    for i in range(0 , min(len(unique_users), 2000), 40):
        item_list = []
        indices = []
        end = min(i+40, 2000)
        users = unique_users[i: end]


        # filtered_numbers = np.array([num for num in unique_items if num not in user_item_train_dict[unique_users[i]]])
        for j in range(len(users)):
            filtered_numbers = np.array([num for num in unique_items])
            item_list.append(filtered_numbers)
            filtered_indices = np.array([num for num, item in enumerate(unique_items) if num in user_item_train_dict[unique_users[j]]])
            indices.append(filtered_indices)

        with th.no_grad():
        # counter += 1

            # graph_dataset = UserItemDataset_2(user_item_graph=test_rank_graph,
            #                                 users = unique_users[i], unique_items = filtered_numbers,
            #                                 original_pair_labales = lab_pair,
            #                                 keyword_edge_cooc_matrix=None,
            #                                 keyword_edge_min_cooc=None, 
            #                                 sample_ratio=1.0, max_nodes_per_hop=200)

            graph_dataset = UserItemDataset(user_item_graph=test_rank_graph,
                                    users = users, unique_items = item_list,
                                    flag = 1,
                                    original_pair_labales = lab_pair,
                                    keyword_edge_cooc_matrix=None,
                                    keyword_edge_min_cooc=None, 
                                    sample_ratio=1.0, max_nodes_per_hop=200)
        
            graph_loader = th.utils.data.DataLoader(graph_dataset, batch_size=len(item_list[0]), shuffle=False, 
                                            num_workers=12, collate_fn=collate_data, pin_memory=True)   
        

        # if additional_feature is not None:
            # graph_loader = th.utils.data.DataLoader(graph_dataset, batch_size=128, shuffle=False, 
            #                                         num_workers=12, collate_fn=collate_data, pin_memory=True, persistent_workers=True)

            
            
            user_predictions = []
            user_labels = []
            # src_node_ids = []
            # dst_node_ids = []
            # counter = end
            indices_counter = 0
            for iter_idx, batch in enumerate(graph_loader, start=1):
                counter += 1
                filtered_indices = indices[indices_counter]
                indices_counter += 1
                # filtered_numbers = np.array([num for num in unique_items if num not in user_item_train_dict[unique_users[counter]]])

                inputs = batch[0].to(args.device)
                labels = inputs.edata['etype'].to(args.device)
                labels = batch[1].to(args.device)
                # target_edges_mask = inputs.edata['is_target_edge'] == 1

                # # Get the src and dst indices of the filtered edges
                # src_target_nodes = inputs.edata['original_src_idx'][target_edges_mask]
                # dst_target_nodes = inputs.edata['original_dst_idx'][target_edges_mask] 
                # target_predictions, count = model(inputs)
                # inputs = batch[0].to(args.device)
                if model_1 is not None:
                    pred, _, emb1 = model_1(inputs)
                    pred, _, emb2 = model_2(inputs)
                    loss, target_predictions = model(emb1, emb2, inputs, tau=0.5)
                else:
                    target_predictions, _,_ = model(inputs)

                # indices = [index for index, item in enumerate(unique_items) if item not in user_item_train_dict[unique_users[counter]]]
                # user_predictions.extend(target_predictions)
                # user_labels.extend(labels)
                target_predictions = [item for idx, item in enumerate(target_predictions) if idx not in filtered_indices]
                labels = [item for idx, item in enumerate(labels) if idx not in filtered_indices]
                user_predictions = th.tensor(target_predictions)
                user_labels = th.tensor(labels)
                ndcg, mrr, new_ndcg = compute_ndcg_mrr_2(user_predictions, user_labels, top_k)

                all_ndcgs.append(ndcg.to(args.device))
                all_mrrs.append(mrr.to(args.device))
                all_base_ndcg.append(new_ndcg.to(args.device))
                

            mean_ndcgs = th.stack(all_ndcgs)
            mean_mrr = th.stack(all_mrrs)
            # mean_base_ndcg = th.stack(all_base_ndcg)
            # all_ndcgs = th.stack(all_ndcgs)
            print("ndcg mean: ", th.mean(mean_ndcgs))
            print("all_mrrs mean: ", th.mean(mean_mrr))
            # print("all base ndcg: ", th.mean(mean_base_ndcg))
                    # print("all_ndcgs",all_ndcgs )

                        # all_ndcgs_cpu = all_ndcgs.cpu().numpy()
                        # all_mrrs_cpu = all_mrrs.cpu().numpy()
    all_ndcgs = th.stack(all_ndcgs)
    all_mrrs = th.stack(all_mrrs)
    all_base_ndcg = th.stack(all_base_ndcg)

        # Now calculate the mean
    return th.mean(all_ndcgs), th.mean(all_mrrs), th.mean(all_base_ndcg)


