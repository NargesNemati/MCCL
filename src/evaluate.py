import math

import torch as th
import numpy as np
import torch.nn as nn
from torch import optim
import pandas as pd

import time
from easydict import EasyDict

from utils import get_logger, get_args_from_yaml, evaluate

from dataloader_v2 import get_dataloader, get_graphs

# from models.egmc import EGMC
from models.igmc import IGMC
from models.kgmc import KGMC
NUM_WORKER = 16
def test(args:EasyDict, logger):
    print(args)
    th.manual_seed(0)
    np.random.seed(0)

    data_path = f'../data/{args.data_name}/{args.data_name}'
    item_cooc_edge_df, user_cooc_edge_df, user_item_cooc_edge_df = None, None, None 
    if args.item_cooc_edge_df is not None :
        item_cooc_edge_df = pd.read_csv(f'../data/{args.data_name}/{args.item_cooc_edge_df}_cooc.csv', index_col=0) 
    if args.user_cooc_edge_df is not None :
        user_cooc_edge_df = pd.read_csv(f'../data/{args.data_name}/{args.user_cooc_edge_df}_cooc.csv', index_col=0) 
    if args.user_item_cooc_edge_df is not None :
        user_item_cooc_edge_df = pd.read_csv(f'../+data/{args.data_name}/{args.user_item_cooc_edge_df}_cooc.csv', index_col=0) 
        

    train_graph, valid_graph, test_graph = get_graphs(data_path=data_path, 
                                                      item_cooc_df=item_cooc_edge_df, 
                                                      user_cooc_df=user_cooc_edge_df, 
                                                      user_item_cooc_df=user_item_cooc_edge_df)
    

    test_loader = get_dataloader(test_graph, 
                                 batch_size=args.batch_size, 
                                 num_workers=NUM_WORKER, 
                                 shuffle=False,
                                 )

    if args.model_type == 'KGMC': 
 
        in_feats = (args.hop+1)*2      
        model = KGMC(in_feats=in_feats, 
                     latent_dim=args.latent_dims,
                     num_relations=5, 
                     num_bases=4, 
                     regression=True,
                     edge_dropout=args.edge_dropout,
                     ).to(args.device)
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")


    model.load_state_dict(th.load(f"../parameters/{args.parameters}"))
    model.to(args.device)

    logger.info("Loading network finished ...\n")
    test_rmse = evaluate(model, test_loader, args.device)
    return test_rmse
    
import yaml

def main():
    with open('../test_configs/test_list.yaml') as f:
        files = yaml.load(f, Loader=yaml.FullLoader)
    file_list = files['files']
    for f in file_list:
        args = get_args_from_yaml(f)
        logger = get_logger(name=args.key, path=f"{args.log_dir}/{args.key}.log")
        # for data_name in args.datasets:
        sub_args = args
        data_name = 'movie'
        sub_args['data_name'] = data_name
        best_rmse_list = []
        test_rmse = test(args, logger=logger)
        logger.info(f"DATASET : {sub_args['data_name']}")
        logger.info(f"Testing RMSE is {test_rmse:.6f}")

if __name__ == '__main__':
    main()