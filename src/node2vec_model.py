# -*- coding: utf-8 -*-
"""
@author: zgz
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import autograd
from torch_geometric.data import DataLoader
from torch_geometric import utils
from torch.optim import lr_scheduler
from torch.utils.data import random_split
from torch_geometric.nn import Node2Vec

import argparse
import os
import time
import setproctitle
import sys
import logging
import copy
from functools import reduce

from dataset_processing import *
from gcn_model_global import *
from utils import *

torch.autograd.set_detect_anomaly(True)

def setup_logging(args):
    # 清空/创建文件
    with open(args.log_file,'w') as file:
        pass
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler(args.log_file)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def init_model(args, id_embeddings):
    # model
    model = N2V(args, id_embeddings).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = None

    print(model)
    train_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    print('Trainable Parameters:', np.sum([p.numel() for p in train_params]))

    return (model, optimizer, scheduler)


def nan_hook(self, inp, output):
    for i, out in enumerate(output):
        nan_mask = torch.isnan(out)
        if nan_mask.any():
            print("In", self.__class__.__name__)
            raise RuntimeError(f"Found NAN in output {i} at indices: ", nan_mask.nonzero(), "where:",
                               out[nan_mask.nonzero()[:, 0].unique(sorted=True)])


def train(data, train_loaders, valid_id):
    data = data.to(args.device)
    model.train()
    start = time.time()
    min_loss = 1e5
    patience = 0
    for epoch in range(args.epochs):
        print('Epoch {}:'.format(epoch))
        train_loss = 0.
        num_iters = len(train_loaders)
        for batch_idx, train_ids in enumerate(train_loaders):
            optimizer.zero_grad()
            out = model(data)
            loss = F.mse_loss(out[train_ids], data.y[train_ids], reduction='sum')
            train_loss += F.l1_loss(out[train_ids], data.y[train_ids], reduction='sum').item()
            with torch.autograd.detect_anomaly():
                loss.backward()
            optimizer.step()
            time_iter = time.time() - start
        train_loss /= num_train
        print("FOLD {}, Time {:.4f} -- Training loss:{}".format(fold, time_iter, train_loss))
        val_loss = test(model, data, valid_id)
        print("FOLD {}, Time {:.4f} -- Validation loss:{}".format(fold, time_iter, val_loss))
        if val_loss < min_loss:
            torch.save(model.state_dict(),
                        '../model/n2vmodel_{}_{}_{}_{}_{}_{}'.format(args.model, args.lr, args.weight_decay,
                                                        args.n_hidden, args.batch_size, args.n_embedding))
            print("!!!!!!!!!! Model Saved !!!!!!!!!!")
            min_loss = val_loss
            patience = 0
        else:
            patience += 1
        if patience > args.patience:
            break


def test(model, data, test_ids):
    model.eval()
    start = time.time()
    data = data.to(args.device)
    out = model(data)
    mae = F.l1_loss(out[test_ids], data.y[test_ids], reduction='mean')
    return mae.item()


def generate_combination(l1,l2):
    res = []
    for u in l1:
        for v in l2:
            if type(u) is not list:
                u = [u]
            if type(v) is not list:
                v = [v]
            res.append(u+v)
    return res


def generate_grid_search_params(search_params):
    if len(search_params.keys()) == 1:
        return search_params.values()
    else:
        return reduce(generate_combination, search_params.values())



if __name__ == '__main__':
    # Experiment parameters
    parser = argparse.ArgumentParser(description='Graph convolutional networks for influencer value prediction')
    parser.add_argument('-sd', '--seed', type=int, default=630, help='random seed')
    parser.add_argument('-lr', '--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('-b', '--batch_size', type=int, default=100, help='batch size')
    parser.add_argument('-wd', '--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('-e', '--epochs', type=int, default=400, help='number of epochs')
    parser.add_argument('-d', '--dropout_ratio', type=float, default=0.5, help='dropout rate')
    parser.add_argument('-dvs', '--device', type=str, default='cuda:0')
    parser.add_argument('-m', '--model', type=str, default='gcn_el', help='model')
    parser.add_argument('-dp', '--dataset_path', type=str, default='../data/sample2_dataset_norm.npy',
                        help='node feature matrix data path')
    parser.add_argument('-nh', '--n_hidden', type=int, default=32, help='number of hidden nodes in each layer of GCN')
    parser.add_argument('-pr', '--pooling_ratio', type=int, default=0.5, help='Pooling ratio for Pooling layers')
    parser.add_argument('-p', '--patience', type=int, default=150, help='Patience')
    parser.add_argument('-fnf', '--full_node_feature', type=int, default=0,
                        help='whether to include #neighbor as featrue')
    parser.add_argument('--n_id_embedding', type=int, default=5, help='id embedding size')
    parser.add_argument('--n_embedding', type=int, default=20, help='embedding size')
    parser.add_argument('--n_folds', type=int, default=10, help='n_folds')
    parser.add_argument('--pretrain_flag', type=int, default=1, help='whether pretrain')
    parser.add_argument('--finetune', type=int, default=0, help='whether finetune')
    parser.add_argument('--grid_search', type=int, default=0, help='whether grid_search')
    parser.add_argument('--grid_search_params', type=str, default='', help='grid search params')
    parser.add_argument('--log_file', type=str, default='../log/test.log', help='grid search params')
    parser.add_argument('-wl', '--walk_length', type=int, default=8, help='walk_length')
    parser.add_argument('-cs', '--context_size', type=int, default=3, help='context_size')
    parser.add_argument('-wpn', '--walks_per_node', type=int, default=10, help='walks_per_node')
    parser.add_argument('-ie', '--id_embedding_epochs', type=int, default=400, help='id_embedding_epochs')
    parser.add_argument('-plr', '--pretrain_lr', type=float, default=0.01, help='pretrain_lr')
    parser.add_argument('--n2v_p', type=int, default=1, help='n2v_p')
    parser.add_argument('--n2v_q', type=int, default=1, help='n2v_q')

    args = parser.parse_args()

    # 对args做一些处理
    args.rnd_state = np.random.RandomState(args.seed)
    if args.full_node_feature==1:
        args.n_demographic = 9
    else:
        args.n_demographic = 8
    args_printer(args)

    # 设定相关信息
    torch.manual_seed(args.seed)
    if torch.cuda.is_available:
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True  # 每次训练得到相同结果
    # torch.backends.cudnn.benchmark = True   # 自动优化卷积实现算法
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device
    setproctitle.setproctitle('Influencer@zhangguozhen')  # 设定程序名

    logger = setup_logging(args)

    start_time = time.time()

    ############################### Load Data ###############################
    print('------------------------- Loading data -------------------------')
    dataset = create_dataset_global(os.path.join('..', 'data', args.dataset_path), args.full_node_feature)
    args.num_features = dataset.num_features
    args.num_edge_features = dataset.num_edge_features
    args.num_communities = int(dataset.community.max().item() + 1)
    args.num_nodes = dataset.x.size(0)
    print(args.num_communities)
    train_ids, test_ids = split_train_test(args.num_communities, args.n_folds, args.rnd_state)

    if args.grid_search:
        logger.info('Start grid_search')
        search_params = eval(args.grid_search_params)

        print('------------------------- Pre-train id embedding -------------------------')
        id_embedding_model = Node2Vec(args.num_nodes, args.n_id_embedding, 
            walk_length=args.walk_length, context_size=args.context_size, 
            walks_per_node=args.walks_per_node, device=args.device)

        if args.pretrain_flag:
            pretrain_id_embedding(id_embedding_model, dataset, 100)
        else:
            id_embedding_model.load_state_dict(torch.load('../model/pretrain_model'))

        id_embeddings = id_embedding_model(torch.arange(args.num_nodes))
        id_embeddings.detach_()
        if args.finetune:
            id_embeddings = torch.nn.Parameter(id_embeddings)
        print('------------------------- Done! -------------------------')

        best_params = []
        best_acc_folds = []
        best_acc = 1e5
        for gs_params in generate_grid_search_params(search_params):

            for i,key in enumerate(search_params.keys()):
                exec('args.'+key+'=gs_params[i]')

            acc_folds = []
            for fold in range(args.n_folds):
                train_loaders, num_train = make_batch(train_ids[fold], args.batch_size, args.rnd_state)

                print('\nFOLD {}, train {}, test {}'.format(fold, num_train, len(test_ids[fold])))

                print('\n------------------------- Initialize Model -------------------------')
                model, optimizer, scheduler = init_model(args)

                print('\n------------------------- Training -------------------------')
                train(dataset, train_loaders, test_ids[fold])

                print('\n------------------------- Testing -------------------------')
                model.load_state_dict(torch.load('../model/model_{}_{}_{}_{}_{}_{}'.format(args.model, args.lr, args.weight_decay,
                                                                                            args.n_hidden, args.batch_size, args.n_embedding)))
                test_loss = test(model, dataset, test_ids[fold])
                acc_folds.append(test_loss)
                
                print('---------------------------------------')
                print('test_loss: {}'.format(test_loss))

            args_printer(args)

            logger.info('model:%s, n_hidden:%d, lr:%f, weight_decay:%f, batch_size:%d, n_embedding:%d',
                        args.model, args.n_hidden, args.lr, args.weight_decay, args.batch_size, args.n_embedding)
            logger.info('acc_folds: %s', str(acc_folds))
            logger.info('%d-fold cross validation avg acc (+- std): %f (%f)', args.n_folds, np.mean(acc_folds), np.std(acc_folds))
            logger.info('---------------------------------------------------------')

            if np.mean(acc_folds)<best_acc:
                best_acc = np.mean(acc_folds)
                best_acc_folds = acc_folds
                best_params = gs_params

        logger.info('Search Done!')
        logger.info('best acc_folds: %s', str(best_acc_folds))
        logger.info('avg acc (+- std): %f (%f)', np.mean(best_acc_folds), np.std(best_acc_folds))
        for i, key in enumerate(search_params.keys()):
            logger.info('Best parameters: %s:%s', str(key), str(best_params[i]))
        print('Search Done!')
        print('Total search time: {}', time.time()-start_time)
        print('best acc_folds: {}'.format(str(best_acc_folds)))
        print('avg acc (+- std): {} ({})'.format(np.mean(best_acc_folds), np.std(best_acc_folds)))
        for i, key in enumerate(search_params.keys()):
            print('Best parameters: {}:{}'.format(str(key), str(best_params[i])))

    # 不grid_search的情况
    else:
        print('------------------------- Pre-train id embedding -------------------------')
        id_embedding_model = Node2Vec(args.num_nodes, args.n_id_embedding, 
            walk_length=args.walk_length, context_size=args.context_size, 
            walks_per_node=args.walks_per_node, p=args.n2v_p, q=args.n2v_q)

        if args.pretrain_flag:
            id_embeddings = pretrain_id_embedding(id_embedding_model, args, dataset)
        else:
            id_embedding_model.load_state_dict(torch.load('../model/pretrain_model_{}_{}'.format(args.walk_length, 
                args.context_size)))
            id_embedding_model = id_embedding_model.to(args.device)
            id_embeddings = id_embedding_model(torch.arange(args.num_nodes).to(args.device))
            
        id_embeddings.detach_()

        if args.finetune:
            id_embeddings = torch.nn.Parameter(id_embeddings)
        print('------------------------- Done! -------------------------')

        acc_folds = []
        for fold in range(args.n_folds):
            train_loaders, num_train = make_batch(train_ids[fold], args.batch_size, args.rnd_state)

            print('\nFOLD {}, train {}, test {}'.format(fold, num_train, len(test_ids[fold])))

            print('\n------------------------- Initialize Model -------------------------')
            model, optimizer, scheduler = init_model(args, id_embeddings)

            print('\n------------------------- Training -------------------------')
            train(dataset, train_loaders, test_ids[fold])

            print('\n------------------------- Testing -------------------------')
            model.load_state_dict(torch.load('../model/n2vmodel_{}_{}_{}_{}_{}_{}'.format(args.model, args.lr, 
                args.weight_decay, args.n_hidden, args.batch_size, args.n_embedding)))

            test_loss = test(model, dataset, test_ids[fold])
            acc_folds.append(test_loss)
            
            print('---------------------------------------')
            print('test_loss: {}'.format(test_loss))

        args_printer(args)
        print(acc_folds)
        print('Total train time: {}', time.time()-start_time)
        print('{}-fold cross validation avg acc (+- std): {} ({})'.format(args.n_folds, np.mean(acc_folds), np.std(acc_folds)))
