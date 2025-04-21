'''
evaluate result with hitrate & NDCG
'''
import warnings
warnings.filterwarnings(action='ignore')

import torch as th
import numpy as np
import pandas as pd
import math

from easydict import EasyDict

from utils import get_logger, get_args_from_yaml

from dataloader_v2 import get_dataloader, get_graphs
from models.igmc import IGMC

def get_prediction(model, loader, device):
    model.eval()
    predicted_values = []
    for batch in loader:
        with th.no_grad():
            preds = model(batch[0].to(device))
        predicted_values += preds.tolist()
    return predicted_values

def evaluate_hitrate(df, predicted_score, k):
    df['score'] = predicted_score
    uids = set(df.user_id)
    hit_count = 0
    for uid in uids:
        sub_df = df.query(f'user_id=={uid}')
        sub_df.sort_values('score', ascending=False, inplace=True)
        sub_df = sub_df[:k]
        hit_count += sum(sub_df.rating) # Sum is same as count hit
    hr = hit_count/len(uids)
    return hr

def evaluate_ndcg(df, predicted_score, k):
    df['score'] = predicted_score
    uids = set(df.user_id)
    ndcg = 0
    for uid in uids:
        sub_df = df.query(f'user_id=={uid}')
        sub_df.sort_values('score', ascending=False, inplace=True)
        sub_df = sub_df[:k].reset_index()
        try:
            position = sub_df.query('rating==1.').index[0]
            ndcg += 1/math.log2(position+2)
        except:
            pass
    ndcg = ndcg/len(uids)
    return ndcg

def evaluate_mrr(df, predicted_score, k):
    df['score'] = predicted_score
    uids = set(df.user_id)
    mrr = 0
    for uid in uids:
        sub_df = df.query(f'user_id=={uid}')
        sub_df.sort_values('score', ascending=False, inplace=True)
        sub_df = sub_df[:k].reset_index()
        try:
            position = sub_df.query('rating==1.').index[0]
            mrr += 1/(position+1) # Make position start to 1
        except:
            pass
    mrr = mrr/len(uids)
    return mrr

K = 10
NUM_WORKER = 16

def test(args:EasyDict, logger):
    th.manual_seed(0)
    np.random.seed(0)

    data_path = f'data/{args.data_name}/{args.data_name}'
    item_cooc_edge_df, user_cooc_edge_df, user_item_cooc_edge_df = None, None, None 
    if args.item_cooc_edge_df is not None :
        item_cooc_edge_df = pd.read_csv(f'data/{args.data_name}/{args.item_cooc_edge_df}_cooc.csv', index_col=0) 
    if args.user_cooc_edge_df is not None :
        user_cooc_edge_df = pd.read_csv(f'data/{args.data_name}/{args.user_cooc_edge_df}_cooc.csv', index_col=0) 
    if args.user_item_cooc_edge_df is not None :
        user_item_cooc_edge_df = pd.read_csv(f'data/{args.data_name}/{args.user_item_cooc_edge_df}_cooc.csv', index_col=0) 
        

    _, _, test_graph = get_graphs(data_path=data_path, 
                                item_cooc_df=item_cooc_edge_df, 
                                user_cooc_df=user_cooc_edge_df, 
                                user_item_cooc_df=user_item_cooc_edge_df)



    #set user-item pairs with negative samples
    hr_test_df = pd.read_csv(f'data/{args.data_name}/{args.data_name}_test_hr.csv', index_col=0)
    uids, iids, labels = hr_test_df.user_id, hr_test_df.item_id, hr_test_df.rating
    test_graph.set_user_item_pairs(uids, iids, labels)  

    test_loader = get_dataloader(test_graph, 
                                 batch_size=args.batch_size, 
                                 num_workers=NUM_WORKER, 
                                 shuffle=False,
                                 )

    if args.model_type == 'IGMC':
        model = IGMC(in_feats=4, 
                     latent_dim=args.latent_dims,
                     num_relations=args.num_relations, 
                     num_bases=4, 
                     regression=True,
                     edge_dropout=args.edge_dropout,
                     ).to(args.device)


    model.load_state_dict(th.load(f"../parameters/{args.parameters}"))
    model.to(args.device)

    logger.info("Loading network finished ...\n")

    hr_test_df.rating = (hr_test_df.rating - 1)/4
    predicted_scores = get_prediction(model, test_loader, args.device)
    test_hitrate = evaluate_hitrate(hr_test_df, predicted_scores, K)
    test_ndcg = evaluate_ndcg(hr_test_df, predicted_scores, K)
    return test_hitrate, test_ndcg
    
import yaml

def main():
    with open('./test_configs/test_list.yaml') as f:
        files = yaml.load(f, Loader=yaml.FullLoader)
    file_list = files['files']
    for f in file_list:
        args = get_args_from_yaml(f)
        logger = get_logger(name=args.key, path=f"{args.log_dir}/{args.key}.log")
        for data_name in args.datasets:
            sub_args = args
            sub_args['data_name'] = data_name
            test_hitrate, test_ndcg = test(args, logger=logger)
            logger.info(f"DATASET : {sub_args['data_name']}")
            logger.info(f"Testing HitRate@{K} is {test_hitrate:.6f}")
            logger.info(f"Testing NDCG@{K} is {test_ndcg:.6f}")

if __name__ == '__main__':
    main()