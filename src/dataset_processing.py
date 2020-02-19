# -*- coding: utf-8 -*-
"""
@author: zgz
"""
import torch
import torch.utils
import torch.utils.data
import numpy as np
import copy
import collections

import pickle
from torch_geometric.data import Data


def create_dataset(path):
    data_samples = np.load(path, allow_pickle=True)
    max_community_size = max([len(u[0]) for u in data_samples])
    dataset = []
    for data in data_samples:
        x = torch.tensor(data[0], dtype=torch.float)
        x = x[:,:-1]
        edge_index = torch.tensor(data[1], dtype=torch.long)
        edge_index = edge_index.t().contiguous()
        edge_attr = torch.tensor(data[2], dtype=torch.float)
        y = torch.tensor([data[-1]], dtype=torch.float)
        dataset.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y))
    return dataset

        
def create_dataset_id(path):
    data_samples = np.load(path, allow_pickle=True)
    max_community_size = max([len(u[0]) for u in data_samples])
    dataset = []
    for data in data_samples:
        x = torch.tensor(data[0], dtype=torch.float)
        x = x[:,:-1]
        community_size = len(data[0])
        ids = torch.eye(community_size)
        ids = torch.cat([ids, torch.zeros(community_size, max_community_size-community_size)], dim=1)
        edge_index = torch.tensor(data[1], dtype=torch.long)
        edge_index = edge_index.t().contiguous()
        y = torch.tensor([data[2]], dtype=torch.float)
        dataset.append(Data(x=x, edge_index=edge_index, y=y, ids=ids))
    return dataset


def get_adj_nodes(edges):
    adj_nodes = collections.defaultdict(list)
    for u in edges:
        adj_nodes[u[0]] += [u[1]]
    return adj_nodes

def get_node_degree(adj_nodes):
    nodes_degree = {}
    for key in adj_nodes.keys():
        nodes_degree[key] = len(adj_nodes[key])
    return nodes_degree


# 社群：重要节点
def get_top_degree_nodes(nodes_degree, community, ratio=0.2):
    top_degree_nodes = {}
    for agent, members in community.items():
        topN = int(len(members)*ratio)
        tem = [(m, nodes_degree[m]) for m in members]
        tem = sorted(tem, key=lambda x:x[1])
        tem = tem[:topN]
        top_degree_nodes[agent] = [u[0] for u in tem]
    return top_degree_nodes


def get_pooling_edges(pooling_nodes, community):
    community_idx = torch.tensor([], dtype=torch.float)
    edges_for_pooling = torch.tensor([], dtype=torch.float)
    for agent, nodes_source in pooling_nodes.items():
        nodes_destination = community[agent]
        num_s = len(nodes_source)
        num_d = len(nodes_destination)
        nodes_source = torch.tensor(nodes_source, dtype=torch.float).reshape((-1,1))
        nodes_source = nodes_source*torch.ones(1,num_d)
        nodes_source = nodes_source.reshape((1,-1))
        nodes_destination = torch.tensor(nodes_destination, dtype=torch.float)
        nodes_destination = nodes_destination.repeat(1,num_s)
        edges = torch.cat((nodes_source, nodes_destination), dim=0)
        community_index = torch.tensor([agent], dtype=torch.float)*torch.ones(1,num_s*num_d)

        edges_for_pooling = torch.cat((edges_for_pooling, edges), dim=1)
        community_idx = torch.cat((community_idx, community_index), dim=1)

    return edges_for_pooling.contiguous().long(), community_idx.contiguous().long()

def create_dataset_DW(path):
    data = np.load(path,allow_pickle = True)
    x = torch.tensor(data,dtype=torch.float)
    return x

def create_dataset_global(path, full_node_feature=0):
    data_samples = np.load(path, allow_pickle=True)
    dataset = []
    x = torch.tensor(data_samples[0], dtype=torch.float)
    if not full_node_feature:
        x = x[:, :-1]
    else:
        x = x[:, torch.tensor([i for i in range(8)]+[-1]+[i for i in range(8,20)])]
    num_data = x.size()[0]
    # ids = torch.eye(num_data)
    edge_index = torch.tensor(data_samples[1], dtype=torch.long)
    edge_index = edge_index.t().contiguous()
    edge_attr = torch.tensor(data_samples[2], dtype=torch.float)
    y = torch.tensor(data_samples[3], dtype=torch.float)
    
    # 需要根据edge恢复一个每个节点的邻接矩阵，给出每个社群中度最大的节点, 根据度最大的点构建一个全连接图
    adj_nodes = get_adj_nodes(data_samples[1])
    nodes_degree = get_node_degree(adj_nodes)
    pooling_nodes = get_top_degree_nodes(nodes_degree, data_samples[-1], 0.2)
    pooling_edges, community_idx_for_pooling = get_pooling_edges(pooling_nodes, data_samples[-1])

    # 需要把在同一个社群的边和不在同一个社群的边区分开来
    # 最终得到adj_inter, edge_attr_inter 和 adj_intra, edge_attr_intra
    community_partition = {}
    for u,v in adj_nodes.items():
        community_partition[u] = set(v)
    adj_intra = []
    edge_attr_intra = []
    adj_inter = []
    edge_attr_inter = []
    for i, edge in enumerate(data_samples[1]):
        if community_partition[edge[0]] & community_partition[edge[1]]:
            adj_inter.append([edge[0],edge[1]])
            edge_attr_inter.append(data_samples[2][i])
        else:
            adj_intra.append([edge[0],edge[1]])
            edge_attr_intra.append(data_samples[2][i])
    adj_intra = torch.tensor(adj_intra, dtype=torch.long)
    adj_intra = adj_intra.t().contiguous()
    adj_inter = torch.tensor(adj_inter, dtype=torch.long)
    adj_inter = adj_inter.t().contiguous()
    edge_attr_intra = torch.tensor(edge_attr_intra, dtype=torch.float)
    edge_attr_inter = torch.tensor(edge_attr_inter, dtype=torch.float)

    # 需要把在多个社群的节点算进来
    # 计算在多个社群的节点，并记录下来，生成slicing list和community index
    # community = torch.zeros(num_data).long()
    community = [-1]*num_data
    multi_community_nodes = []
    multi_community_index = []
    for c, members in data_samples[4].items():
        # community[members] = c
        for member in members:
            if community[member]==-1:
                community[member] = c
            else:
                multi_community_nodes.append(member)
                multi_community_index.append(c)
    community = torch.tensor(community).long()
    multi_community_nodes = torch.tensor(multi_community_nodes).long()
    multi_community_index = torch.tensor(multi_community_index).long()
    dataset = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, community=community,
                   pooling_edges=pooling_edges, community_idx_for_pooling=community_idx_for_pooling,
                   multi_community_nodes=multi_community_nodes, multi_community_index=multi_community_index,
                   adj_inter=adj_inter, adj_intra=adj_intra, edge_attr_inter=edge_attr_inter,
                   edge_attr_intra=edge_attr_intra)
    return dataset


# def split_train_test(n, rnd_state=None):
#     rnd_state = np.random.RandomState() if rnd_state is None else rnd_state
#     idx = rnd_state.permutation(n)
#     train_num = int(n*0.8)
#     train_idx = idx[:train_num]
#     test_idx = idx[train_num:]
#     return (train_idx, test_idx)


def split_train_test(n, nfolds, rnd_state=None):
    rnd_state = np.random.RandomState() if rnd_state is None else rnd_state
    idx = rnd_state.permutation(n)
    idx = idx.tolist()
    stride = int(n/nfolds)
    # 先把idx分成10份
    idx = [idx[i*stride:(i+1)*stride] for i in range(nfolds)]
    train_idx, test_idx = {},{}
    for fold in range(nfolds):
        test_idx[fold] = np.array(copy.deepcopy(idx[fold]))
        train_idx[fold] = []
        for i in range(nfolds):
            if i!=fold:
                train_idx[fold] += idx[i]
        train_idx[fold] = np.array(train_idx[fold])
    return train_idx, test_idx


def split_train_valid_test(n, nfolds, rnd_state=None):
    rnd_state = np.random.RandomState() if rnd_state is None else rnd_state
    idx = rnd_state.permutation(n)
    idx = idx.tolist()
    stride = int(n/nfolds)
    # 先把idx分成10份
    idx = [idx[i*stride:(i+1)*stride] for i in range(nfolds)]
    train_idx, valid_idx, test_idx = {},{},{}
    for fold in range(nfolds):
        test_idx[fold] = np.array(copy.deepcopy(idx[fold]))
        valid_idx[fold] = np.array(copy.deepcopy(idx[(fold+1)%nfolds]))
        train_idx[fold] = []
        for i in range(nfolds):
            if i!=fold and i!=(fold+1)%nfolds:
                train_idx[fold] += idx[i]
        train_idx[fold] = np.array(train_idx[fold])
    return train_idx, valid_idx, test_idx

def make_batch(train_ids, batch_size, seed):
    """
    return a list of batch ids for mask-based batch.
    Args:
        train_ids: list of train ids
        batch_size: ~
    Output:
        batch ids, e.g., [[1,2,3], [4,5,6], ...]
    """
    num_nodes = len(train_ids)
    rnd_state = seed
    permuted_idx = rnd_state.permutation(num_nodes)
    # print(permuted_idx[0:10])
    # print('batch----------------')
    permuted_train_ids = train_ids[permuted_idx]
    batches = [permuted_train_ids[i*batch_size:(i+1)*batch_size] for i in range(int(num_nodes/batch_size))]
    if num_nodes%batch_size > 0:
        batches.append(permuted_train_ids[(num_nodes-num_nodes%batch_size):])

    return batches, num_nodes



        