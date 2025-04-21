import math, copy

import dgl
import pandas as pd
import numpy as np
import pickle as pkl

import torch as th
import torch.nn as nn
from torch import optim

import time
from easydict import EasyDict
from torch.utils.tensorboard import SummaryWriter


from utils import get_logger, get_args_from_yaml, evaluate, test_ndcg_dataset, evaluate_user_separately, evaluate_user_separately_2
# from dataloader import get_graphs, get_dataloader
from dataloader_v2 import get_graphs, get_dataloader

from models.kgmc import KGMC
from models.vgae import autoencoder
from models.igmc import IGMC
from models.igmc_bert import IGMC_BERT
from models.attention import KGMC_att
import torch.nn.functional as F
from models.contrastive import ContrastiveLearner, KGMC_SAGE


def train_epoch(model, model_1, model_2, loss_fn, bce_loss_fn, optimizer, optimizer_1, 
                optimizer_2, loader, epoch_idx, device, logger,p , log_interval, train_model=None):

    model.train()
    if model_1 is not None:
        model_1.train()
        model_2.train()
    # if model_2 is not None:
    #     model_2.train()


    epoch_loss = 0.
    iter_loss = 0.
    iter_mse = 0.
    iter_cnt = 0
    iter_dur = []

    for iter_idx, batch in enumerate(loader, start=1):
        t_start = time.time()

        if train_model == 'IGMC_BERT':
            inputs = batch[0].to(device)
            vectors = batch[1].to(device)
            labels = batch[2].to(device)
            preds = model_1(inputs, vectors)

        elif train_model == 'autoencoder':
            inputs = batch[0].to(device)
            all_subg_labels = inputs.edata['etype'].to(device)
            labels = batch[1].to(device)
            labe_edge_mask = inputs.edata['edge_mask']
            preds, all_preds, emb= model(inputs)

        elif train_model == 'kgmc':
            inputs = batch[0].to(device)
            all_subg_labels = inputs.edata['etype'].to(device)
            labels = batch[1].to(device)
            # labe_edge_mask = inputs.edata['edge_mask']
            preds, all_preds, emb= model(inputs)

        elif train_model == 'mccl':
            inputs = batch[0].to(device)
            all_subg_labels = inputs.edata['etype'].to(device)
            labels = batch[1].to(device)
            labe_edge_mask = inputs.edata['edge_mask']
            preds_1, all_preds_1, emb_1 = model_1(inputs)
            preds_2, all_preds_2, emb_2 = model_2(inputs)

            contrast_loss, preds = model(emb_1, emb_2, inputs, tau=0.5)
            # preds = preds.squeeze(1)
            # contrast_loss_2 = contrastive_loss(emb_1.detach(), emb_2, users, items, temp=0.5)
            # contrast_loss = contrastive_loss(emb_1, emb_2, users, items, temp=0.5)
            # contrastive_model = ContrastiveKGMC(model_1, model_2)
            # z_view1, z_view2, preds_view1, preds_view2 = contrastive_model(inputs)
        
        elif train_model == 'kgmc_sage':
            inputs = batch[0].to(device)
            all_subg_labels = inputs.edata['etype'].to(device)
            labels = batch[1].to(device)
            labe_edge_mask = inputs.edata['edge_mask']
            preds, all_preds_1, emb_1 = model(inputs)
        elif train_model == 'ATTENTION':
            inputs = batch[0].to(device)
            all_subg_labels = inputs.edata['etype'].to(device)
            labels = batch[1].to(device)
            labe_edge_mask = inputs.edata['edge_mask']
            preds, all_preds_1, emb_1 = model(inputs)


        if train_model == 'mccl':
            kl_divergence = -0.5 * th.sum(1 + model_1.log_std - model_1.mean.pow(2) - model_1.log_std.exp(), dim=1).mean()
            loss_1 = loss_fn(preds_1, labels)
            loss_2 = loss_fn(preds_2, labels)
            beta = 0.001
            all_subg_labels = all_subg_labels.to(th.float32).clone().detach().to(device)
            all_subg_labels = (all_subg_labels - 1) / 4
            loss_reconstruct = loss_fn(all_preds_1, all_subg_labels)
            optimizer.zero_grad()
            loss = (loss_fn(preds, labels) + 0.001 * contrast_loss + 0.001 * kl_divergence
                          + loss_1 + loss_2 + loss_reconstruct * 0.001)
            loss.backward()
            optimizer.step()



        if train_model == 'kgmc_sage':
            loss = loss_fn(preds, labels)
        if train_model == 'ATTENTION':
            loss = loss_fn(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if train_model == 'kgmc':
            loss = loss_fn(preds, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if train_model == 'autoencoder':
            reconstruction_loss_target = loss_fn(preds, labels)
            loss_1 = loss_fn(preds, labels)

            beta = 0.001
            all_subg_labels = all_subg_labels.to(th.float32).clone().detach().to(device)
            all_subg_labels = (all_subg_labels - 1) / 4
            loss_reconstruct = loss_fn(all_preds, all_subg_labels)
            kl_divergence = -0.5 * th.sum(1 + model.log_std - model.mean.pow(2) - model.log_std.exp(), dim=1).mean()
            loss =  0.001 * kl_divergence + loss_1  + loss_reconstruct * 0.01

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # for name, param in model.named_parameters():
        #     if "projection" in name:
        #         print(f"Gradient for {name}: {param.grad}")

        epoch_loss += loss.item() * preds.shape[0]
        iter_loss += loss.item() * preds.shape[0]
        # print(type((labels*4+1)[0]))
        iter_mse += (((preds *4 + 1) - (labels*4 +1)) ** 2).sum().item()
        # iter_mse += (((((preds +1) / 2) *4 + 1) - (labels*4 +1)) ** 2).sum().item()
        iter_cnt += preds.shape[0]
        iter_dur.append(time.time() - t_start)

        if iter_idx % log_interval == 0:
            logger.debug(f"Iter={iter_idx}, loss={iter_loss/iter_cnt:.4f}, rmse={math.sqrt(iter_mse/iter_cnt):.4f}, time={np.average(iter_dur):.4f}")
            # print(reg_loss)
            iter_loss = 0.
            iter_mse = 0.
            iter_cnt = 0
            
    return epoch_loss / len(loader.dataset)


NUM_WORKER = 16
def train(args:EasyDict, logger, model_type, p):


    data_path = f'../data/{args.data_name}/{args.data_name}'
    
    item_cooc_edge_df, user_cooc_edge_df, user_item_cooc_edge_df = None, None, None 
    if args.item_cooc_edge_df != 'None' :
        item_cooc_edge_df = pd.read_csv(f'../data/{args.data_name}/{args.item_cooc_edge_df}_cooc.csv', index_col=0) 
    if args.user_cooc_edge_df != 'None' :
        user_cooc_edge_df = pd.read_csv(f'../data/{args.data_name}/{args.user_cooc_edge_df}_cooc.csv', index_col=0) 
    if args.user_item_cooc_edge_df != 'None' :
        user_item_cooc_edge_df = pd.read_csv(f'../data/{args.data_name}/{args.user_item_cooc_edge_df}_cooc.csv', index_col=0) 
    


    train_graph, valid_graph, test_graph = get_graphs(data_path=data_path, 
                                                      item_cooc_df=item_cooc_edge_df, 
                                                      user_cooc_df=user_cooc_edge_df, 
                                                      user_item_cooc_df=user_item_cooc_edge_df)
    
    train_loader = get_dataloader(train_graph, 
                                 batch_size=args.batch_size, 
                                 num_workers=NUM_WORKER,
                                 shuffle=True, 
                                 )
    valid_loader = get_dataloader(valid_graph, 
                                 batch_size=args.batch_size, 
                                 num_workers=NUM_WORKER, 
                                 shuffle=False,
                                 )
    test_loader = get_dataloader(test_graph, 
                                 batch_size=args.batch_size, 
                                 num_workers=NUM_WORKER, 
                                 shuffle=False,
                                 )

    test_dataset_for_ndcg = test_ndcg_dataset(args)
    # test_dataset_for_rel_ndcg = evaluate_user_separately_2(args)

    ### prepare data and set model
    in_feats = (args.hop+1)*2 
 
    if model_type == 'mccl':
        print("model is mccl")
        model_1 = autoencoder(in_feats=in_feats, 
                     latent_dim=args.latent_dims,
                     num_relations=args.num_relations, 
                     num_bases=None, 
                     regression=True,
                     edge_dropout=args.edge_dropout,
                     ).to(args.device)

        # model_1 = KGMC_SAGE(in_feats=in_feats, 
        #              latent_dim=args.latent_dims,
        #              regression=True,
        #              edge_dropout=args.edge_dropout,
        #              ).to(args.device)
        
        model_2 = KGMC_att(in_feats=in_feats, 
                     latent_dim=args.latent_dims,
                     num_relations=args.num_relations, 
                     num_bases=None, 
                     regression=True,
                     edge_dropout=args.edge_dropout,
                     ).to(args.device)
        model = ContrastiveLearner(embedding_dim=128).to(args.device)

    elif model_type == 'kgmc_sage':
        print("model is sage")
        model = KGMC_SAGE(in_feats=in_feats, 
                     latent_dim=args.latent_dims,
                     regression=True,
                     edge_dropout=args.edge_dropout,
                     ).to(args.device)
    elif model_type == 'ATTENTION':
            model =KGMC_att(in_feats=in_feats, 
                    latent_dim=args.latent_dims,
                    num_relations=args.num_relations, 
                    num_bases=4, 
                    regression=True,
                    edge_dropout=args.edge_dropout,
                    ).to(args.device)
        
    elif model_type == 'kgmc':
            model = KGMC(in_feats=in_feats, 
                    latent_dim=args.latent_dims,
                    num_relations=args.num_relations, 
                    num_bases=None, 
                    regression=True,
                    edge_dropout=args.edge_dropout,
                    ).to(args.device)
        
    elif model_type == 'autoencoder':
        print("model is autoencoder")
        model = autoencoder(in_feats=in_feats, 
                     latent_dim=args.latent_dims,
                     num_relations=args.num_relations, 
                     num_bases=None, 
                     regression=True,
                     edge_dropout=args.edge_dropout,
                     ).to(args.device)

    loss_fn = nn.MSELoss().to(args.device)
    bce_loss_fn = nn.BCELoss().to(args.device)

    if model_type == 'mccl':
        optimizer = optim.Adam(list(model.parameters()) + list(model_1.parameters()) + list(model_2.parameters()),
                                lr=args.train_lr, weight_decay=args.weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.train_lr, weight_decay=args.weight_decay)
    logger.info("Loading network finished ...\n")

    
    best_epoch = 0
    best_rmse = np.inf
    best_ndcg = np.inf
    best_mrr = np.inf

    logger.info(f"Start training ... learning rate : {args.train_lr}")
    epochs = list(range(1, args.train_epochs+1))

    eval_func_map = {
        'mccl': evaluate,
        'kgmc_sage':evaluate,
        'ATTENTION':evaluate,
        'autoencoder':evaluate,
        # 'IGMC': evaluate,
    }
    eval_func = eval_func_map.get(model_type, evaluate)

    for epoch_idx in epochs:
        logger.debug(f'Epoch : {epoch_idx}')
        if model_type != 'mccl':
            model_1 = None
            model_2 = None
        optimizer_1 = None
        optimizer_2 = None
    
        train_loss = train_epoch(model, model_1, model_2, loss_fn, bce_loss_fn,optimizer, optimizer_1, optimizer_2, train_loader, epoch_idx,
                                 args.device, logger, p, args.log_interval, train_model=model_type)
        # val_rmse = eval_func(model, valid_loader, args.device)
        test_rmse = eval_func(model, model_1, model_2, test_loader, args.device)

        eval_info = {
            'dataset': args.data_name,
            'epoch': epoch_idx,
            'train_loss': train_loss,
            'val_rmse' : -1,
            'test_rmse': test_rmse,
        }
        # writer.add_scalar("Metrics/RMSE", test_rmse, epoch_idx)
        logger.info('=== {} Epoch {}, train loss {:.6f}, val rmse {:.6f}, test rmse {:.6f} ==='.format(*eval_info.values()))
        if epoch_idx % 20 == 0:
            test_ndcg, test_mrr, base_ndcg = evaluate_user_separately(model, model_1, model_2, args, 
                                                                      test_dataset_for_ndcg, top_k=args.ndcg_k)
            print("ndcg_ranking: ", base_ndcg)
            print("mrr_ranking: ", test_mrr)
            
            test_rel_ndcg, test_rel_mrr, base__rel_ndcg = evaluate_user_separately_2(model, model_1, model_2,
                                                                                        args, top_k=args.ndcg_k)

            print("ndcg_rating: ", test_rel_ndcg)
            print("mrr rating: ", test_rel_mrr)
            


        if epoch_idx % args.lr_decay_step == 0:
            for param in optimizer.param_groups:
                param['lr'] = args.lr_decay_factor * param['lr']
            print('lr : ', param['lr'])

        if best_rmse > test_rmse:
            logger.info(f'new best test rmse {test_rmse:.6f} ===')
            best_epoch = epoch_idx
            best_rmse = test_rmse
            best_state = copy.deepcopy(model.state_dict())
            if model_type == 'mccl':
                best_state_1 = copy.deepcopy(model_1.state_dict())
                best_state_2 = copy.deepcopy(model_2.state_dict())


    th.save(best_state, f'../parameters/{args.key}_{args.data_name}_{best_rmse:.4f}.pt')
    if model_type == 'mccl':
        th.save(best_state_1, f'../parameters/model_1_{args.data_name}_{best_rmse:.4f}.pt')
        th.save(best_state_2, f'../parameters/model_2_{args.data_name}_{best_rmse:.4f}.pt')

    logger.info(f"Training ends. The best testing rmse is {best_rmse:.6f} at epoch {best_epoch}")
    logger.info("Evaluating on test dataset...")

    logger.info(f"Training ends. The best testing metrics")
    return best_rmse, best_ndcg, best_mrr
    
import yaml
from collections import defaultdict
from datetime import datetime

def main():
    seed = 42 

    dgl.random.seed(seed)
    th.manual_seed(seed)
    np.random.seed(seed)
    th.cuda.manual_seed(seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False
    
    with open('../train_configs/train_list.yaml') as f:
        files = yaml.load(f, Loader=yaml.FullLoader)
    file_list = files['files']
    for f in file_list:
        date_time = datetime.now().strftime("%Y%m%d_%H:%M:%S")
        args = get_args_from_yaml(f)
        logger = get_logger(name=args.key, path=f"{args.log_dir}/{args.key}.log")
        logger.info('train args')
        for k,v in args.items():
            logger.info(f'{k}: {v}')

        # for data_name in args.datasets:
        sub_args = args
        sub_args['data_name'] = sub_args['dataset']
        logger.info(f"DATASET : {sub_args['data_name']}")

        test_results = defaultdict(list)
        best_lr = None
        sub_args = args
        # sub_args['data_name'] = args.dataset
        best_rmse_list = []
        best_mrr_list = []
        best_ndcg_list = []
        # writer = SummaryWriter(log_dir="../results/experiment")
        # b = [0.01, 0.0025,  0.0075]
        b= [0.01]
        for p in b:
            print("contrast prod : ", p)
            for mdl in args.model_type:
                for lr in args.train_lrs:
                    sub_args['train_lr'] = lr

                    best_rmse, best_ndcg, best_mrr = train(sub_args, logger=logger, model_type=mdl, p=p)
                    in_feats = (args.hop+1)*2 


                    if mdl == 'mccl':
                        model_1 = autoencoder(in_feats=in_feats, 
                                latent_dim=args.latent_dims,
                                num_relations=args.num_relations, 
                                num_bases=4, 
                                regression=True,
                                edge_dropout=args.edge_dropout,
                                ).to(args.device)
                        # model_1 = KGMC_SAGE(in_feats=in_feats, 
                        #         latent_dim=args.latent_dims,
                        #         regression=True,
                        #         edge_dropout=args.edge_dropout,
                        #         ).to(args.device)
                        model_2 =KGMC_att(in_feats=in_feats, 
                                latent_dim=args.latent_dims,
                                num_relations=args.num_relations, 
                                num_bases=4, 
                                regression=True,
                                edge_dropout=args.edge_dropout,
                                ).to(args.device)
                        model = ContrastiveLearner(embedding_dim=128).to(args.device)

                    elif mdl == 'kgmc_sage':

                        print("model is sage")
                        model_1 = KGMC_SAGE(in_feats=in_feats, 
                                    latent_dim=args.latent_dims,
                                    regression=True,
                                    edge_dropout=args.edge_dropout,
                                    ).to(args.device)
                        
                    elif mdl == 'ATTENTION':
                        model =KGMC_att(in_feats=in_feats, 
                                latent_dim=args.latent_dims,
                                num_relations=args.num_relations, 
                                num_bases=4, 
                                regression=True,
                                edge_dropout=args.edge_dropout,
                                ).to(args.device)
                    
                    elif mdl == 'autoencoder':
                        model = autoencoder(in_feats=in_feats, 
                                latent_dim=args.latent_dims,
                                num_relations=args.num_relations, 
                                num_bases=4, 
                                regression=True,
                                edge_dropout=args.edge_dropout,
                                ).to(args.device)
                    elif mdl == 'kgmc':
                        model = KGMC(in_feats=in_feats, 
                                latent_dim=args.latent_dims,
                                num_relations=args.num_relations, 
                                num_bases=None, 
                                regression=True,
                                edge_dropout=args.edge_dropout,
                                ).to(args.device)

                    if mdl == 'mccl':
                        model.load_state_dict(th.load(f"../parameters/{args.key}_{args.data_name}_{best_rmse:.4f}.pt", weights_only=True))
                        model_1.load_state_dict(th.load(f"../parameters/model_1_{args.data_name}_{best_rmse:.4f}.pt", weights_only=True))
                        model_2.load_state_dict(th.load(f"../parameters/model_2_{args.data_name}_{best_rmse:.4f}.pt", weights_only=True))

                    

                    else:
                        model.load_state_dict(th.load(f"../parameters/{args.key}_{args.data_name}_{best_rmse:.4f}.pt", weights_only=True))
                        model_1 = None
                        model_2 = None
                    print("--------------------")
                            
                    test_dataset_for_ndcg = test_ndcg_dataset(args)
                    test_ndcg, test_mrr, base_ndcg = evaluate_user_separately(model, model_1, model_2, args, 
                                                                            test_dataset_for_ndcg, top_k=args.ndcg_k)
                    print("ndcg: ", base_ndcg)
                    print("mrr: ", test_mrr)
                    
                    test_rel_ndcg, test_rel_mrr, base__rel_ndcg = evaluate_user_separately_2(model, model_1, model_2,
                                                                        args, top_k=args.ndcg_k)
                    test_ndcg, test_mrr, base_ndcg = evaluate_user_separately(model, model_1, model_2, args, 
                                                                            test_dataset_for_ndcg, top_k=args.ndcg_k)
                    print("ndcg: ", base_ndcg)
                    print("mrr: ", test_mrr)
                    print("rel ndcg: ", test_rel_ndcg)
                    print("rel mrrr: ", test_rel_mrr)

                    test_results[sub_args['data_name']].append(best_rmse)
                    best_rmse_list.append(best_rmse)
    
        mean_std_dict = dict()
        for dataset, results in test_results.items():
            mean_std_dict[dataset] = [f'{np.mean(results):.4f} ± {np.std(results):.5f}']
        mean_std_df = pd.DataFrame(mean_std_dict)
        mean_std_df.to_csv(f'../results/{args.key}_{date_time}.csv')
        # writer.close()

if __name__ == '__main__':

    main()