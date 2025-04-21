import math

import torch as th
import numpy as np
import torch.nn as nn
from torch import optim
import pandas as pd

import time
from easydict import EasyDict

from utils import get_logger, get_args_from_yaml

from dataloader import get_dataloader
from models.igmc import IGMC

def evaluate(model, loader, device):
    # Evaluate RMSE
    model.eval()
    mse = 0.
    result = []
    for batch in loader:
        with th.no_grad():
            preds = model(batch[0].to(device))
        labels = batch[1].to(device)
        mse += ((preds - labels) ** 2).sum().item()
        result.append(preds)
    result = th.concat(result, dim=0)
    print(result.shape)
    mse /= len(loader.dataset)
    return np.sqrt(mse), result


def test(args:EasyDict, logger):
    th.manual_seed(0)
    np.random.seed(0)

    data_path = f'dataset/{args.dataset}/{args.dataset_filename}'
    if args.efeat_path != None:
        efeat_path = f'dataset/{args.dataset}/{args.efeat_path}'
    else:
        efeat_path = None
    _, _, test_loader =  get_dataloader(data_path, batch_size=args.batch_size, feature_path=efeat_path)

    if args.model_type == 'IGMC':
        in_feats = (args.hop+1)*2 
        model = IGMC(in_feats=in_feats, 
                    latent_dim=args.latent_dims,
                    num_relations=5, 
                    num_bases=4, 
                    regression=True, 
                    edge_dropout=args.edge_dropout,
                    )


    model.load_state_dict(th.load(f"../parameters/{args.parameters}"))
    model.to(args.device)

    logger.info("Loading network finished ...\n")
    test_rmse, result = evaluate(model, test_loader, args.device)


    test_df = pd.read_csv(f'{data_path}_test.csv')
    test_df['pred'] = result.tolist()
    test_df.to_csv(f'{args.model_type}_{args.dataset}.csv')
    return test_rmse
    
import yaml

def main():
    with open('./test_configs/test_list.yaml') as f:
        files = yaml.load(f, Loader=yaml.FullLoader)
    file_list = files['files']
    for f in file_list:
        args = get_args_from_yaml(f)
        logger = get_logger(name=args.key, path=f"{args.log_dir}/{args.key}.log")
        test_rmse = test(args, logger=logger)
        logger.info(f"DATASET : {args.dataset}")
        logger.info(f"Testing RMSE is {test_rmse:.6f}")

if __name__ == '__main__':
    main()