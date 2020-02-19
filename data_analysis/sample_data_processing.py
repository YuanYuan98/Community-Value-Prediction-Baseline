# -*- coding: utf-8 -*-
"""
@author: zgz
"""

import numpy as np
import numba as nb
import networkx as nx
import collections
# from sklearn.preprocessing import StandardScaler
# from sklearn.preprocessing import MinMaxScaler



def network_analysis(nodes, edges):
    '''
    计算每个节点对应的特征(num_triangles, ave_clustering_coef, degree_centrality)，返回array
    '''
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)

    tri = nx.algorithms.cluster.triangles(graph).values()
    tri = sum(list(tri))/3
    ave_clustering_coef = nx.algorithms.cluster.average_clustering(graph)
    degree_cen = nx.algorithms.centrality.degree_centrality(graph).values()
    degree_cen = np.array(list(degree_cen))
    degree_cen = np.sum(1 - degree_cen)
    num_nodes = graph.number_of_nodes()
    degree_cen = degree_cen*(num_nodes-1)/(num_nodes*(num_nodes-2))

    return (tri, degree_cen, ave_clustering_coef)


def rf_processing():
    features = {}
    with open('../data/sample3_features_rf', 'r') as file:
        for line in file:
            temp = eval(line)
            features[temp[0]] = temp[1:]

    labels = {}
    with open('../data/sample3_label', 'r') as file:
        for line in file:
            temp = eval(line)
            labels[temp[0]] = [temp[1]]

    nodes = []
    edges = {}
    with open('../data/sample3_edges', 'r') as file:
        for line in file:
            temp = eval(line)
            nodes.append(temp[0])
            edges[temp[0]] = temp[1]

    community = {}
    with open('../data/sample3_community', 'r') as file:
        for line in file:
            temp = eval(line)
            community[temp[0]] = temp[1]

    for agent, members in community.items():
        set_members = set(members)
        c_nodes = list(set_members)
        c_edges = []
        for u in set_members:
            c_edges += [(u, v) for v in edges[u] if v in set_members]
        tri, degree_cen, ave_clustering_coef = network_analysis(c_nodes, c_edges)
        features[agent] += [None, None, None]
        features[agent][-3:] = [tri, degree_cen, ave_clustering_coef]
        # 顺便处理None数据
        for i in range(len(features[agent])):
            if features[agent][i]==None:
                features[agent][i] = 0

    dataset = []
    for agent in features.keys():
        dataset.append(features[agent] + labels[agent])
    dataset = np.array(dataset)
    
    np.save('../data/sample3_dataset_rf.npy', dataset)

    return dataset


def agent_wise_gcn_processing(feature_path, community_path, label_path, output_path, edge_feature_path=None):
    '''
    Process data for GCN model
    Input:
        features:
        communities
        labels
        output path
        edge features
    Output:
        sample_dataset.npy which contains (feature_matrix, adj_matrix, label[agent])
    '''
    # 0:feature 1:adj_nodes
    node_feature = {}
    with open(feature_path, 'r') as file:
        for line in file:
            temp = line.strip()
            if len(temp) > 5:
                res = temp[1:-3].split('\t')
                for u in res:
                    tem = u.split('|')
                    node_feature[tem[0]] = eval(tem[1])

    # 处理none数据
    for u, v in node_feature.items():
        if v[0][7] is None:
            node_feature[u][0][7] = [0]*12

    # 处理sub_list
    for u, v in node_feature.items():
        node_feature[u] = (node_feature[u][0][:7] + node_feature[u][0][7] + node_feature[u][0][8:], node_feature[u][1])

    # 读入社群
    community_partition = {}
    with open(community_path, 'r') as file:
        for line in file:
            temp = eval(line)
            community_partition[temp[0]] = temp[1]

    # 读入edge_feature
    if edge_feature_path:
        edge_feature = {}
        with open(edge_feature_path, 'r') as file:
            for line in file:
                line = line.strip()
                if len(line)>5:
                    data = line[1:-3].split('\t')
                    for d in data:
                        temp = d.split('|')
                        node_i = temp[0].split(',')[0]
                        node_j = temp[0].split(',')[1]
                        edge = node_i+node_j if node_i<node_j else node_j+node_i
                        edge_feature[edge] = eval(temp[1])

    # 读入label
    label = {}
    with open(label_path, 'r') as file:
        for line in file:
            temp = eval(line)
            label[temp[0]] = temp[1]

    agents = set(community_partition.keys())
    users = set(node_feature.keys())

    # 删除其他还包括None的数据(还是会有None数据，直接删掉，留下剩下的点和对应的边)
    remove_list = []
    for user,features in node_feature.items():
        basic_info = features[0]
        for f in basic_info:
            if f == None:
                remove_list.append(user)
                break
    for user in remove_list:
        users.remove(user)
        if user in agents:
            agents.remove(user)

    # 0:feature_matrix 1:adj_matrix(需要转置) 2:label
    count = 0
    dataset = []
    for agent, fans in community_partition.items():
        feature_matrix = []
        adj_matrix = []
        edge_attr = []
        # 节点映射
        node_projection = {}
        fans = [u for u in fans if u in users]
        for i, fan in enumerate(fans):
            node_projection[fan] = i
        community = set(fans)
        for fan in fans:
            feature_matrix.append(node_feature[fan][0])
            # 需要判断neighbor是否在此社群,从而只保留社群内的边
            for neighbor in node_feature[fan][1]:
                if neighbor in community:
                    adj_matrix.append([node_projection[fan], node_projection[neighbor]])
                    # 如果需要处理edge_feature
                    if edge_feature_path:
                        edge_idx = fan+neighbor if fan<neighbor else neighbor+fan
                        if edge_idx in edge_feature:
                            edge_attr.append(edge_feature[edge_idx])
                        else:
                            edge_attr.append([0]*12)
                            count += 1
        if edge_feature_path:
            dataset.append((feature_matrix, adj_matrix, edge_attr, label[agent]))
        else:
            dataset.append((feature_matrix, adj_matrix, label[agent]))

    print(count)
    print('...................')
    # save dataset for loading
    np.save(output_path, dataset)
    return dataset


def global_gcn_processing(feature_path, community_path, label_path, output_path, edge_feature_path, edge_path, disconnected_flag=False):
    '''
    Process data for global GCN model
    Input:
        features:
        communities
        labels
        output path
        edge features
    Output:
        sample_dataset.npy which contains [feature_matrix, adj_matrix, label_matrix, community_partition_index]
    '''
    # 0:feature 1:adj_nodes
    node_feature = {}
    with open(feature_path, 'r') as file:
        for line in file:
            u = eval(line)
            node_feature[u[0]] = u[1:]

    # 读入社群
    community_partition = {}
    with open(community_path, 'r') as file:
        for line in file:
            u = eval(line)
            community_partition[u[0]] = u[1]

    # 社群归属
    community_assignment = collections.defaultdict(set)
    for community, members in community_partition.items():
        for member in members: 
            community_assignment[member].add(community)

    # 读入edge_feature
    if edge_feature_path:
        edge_feature = {}
        with open(edge_feature_path, 'r') as file:
            for line in file:
                t = eval(line)
                node_i = t[0][0]
                node_j = t[0][1]
                edge = node_i+node_j if node_i<node_j else node_j+node_i
                edge_feature[edge] = t[1:]

    # 读入edges
    adj_nodes = {}
    with open(edge_path, 'r') as file:
        for line in file:
            t = eval(line)
            adj_nodes[t[0]] = t[1]

    # 读入label
    label = {}
    with open(label_path, 'r') as file:
        for line in file:
            t = eval(line)
            label[t[0]] = t[1]

    agents = set(community_partition.keys())
    users = set(node_feature.keys())
    n_agents = len(agents)
    print('number of agents: ', n_agents)

    # # 删除其他还包括None的数据(还是会有None数据，直接删掉，留下剩下的点和对应的边)
    # remove_list = []
    # for user,features in node_feature.items():
    #     basic_info = features[0]
    #     for f in basic_info:
    #         if f == None:
    #             remove_list.append(user)
    #             break
    # for user in remove_list:
    #     users.remove(user)
    #     if user in agents:
    #         agents.remove(user)

    # 节点映射
    node_projection = {}
    for index,agent in enumerate(agents):
        node_projection[agent] = index
    fans = users.difference(agents)
    for index,user in enumerate(fans):
        node_projection[user] = index + n_agents

    # 0:feature_matrix 1:adj_matrix(需要转置) 2:label
    dataset = []
    feature_matrix = []
    adj_matrix = []
    label_matrix = []
    edge_attr = []
    community_partition_index = {}

    for agent in agents:
        label_matrix.append((node_projection[agent], label[agent]))
    label_matrix = sorted(label_matrix)
    label_matrix = list(map(lambda x: x[1], label_matrix))

    for user in users:
        feature_matrix.append((node_projection[user], node_feature[user]))
        for neighbor in adj_nodes[user]:
            if neighbor in node_projection:
                if disconnected_flag: # test：只筛选在同一个社群的点
                    if community_assignment[user] & community_assignment[neighbor]:
                        adj_matrix.append([node_projection[user], node_projection[neighbor]])
                        edge_idx = user+neighbor if user<neighbor else neighbor+user
                        if edge_idx in edge_feature:
                            edge_attr.append(edge_feature[edge_idx])
                        else:
                            edge_attr.append([0]*12)
                else:
                    adj_matrix.append([node_projection[user], node_projection[neighbor]])
                    edge_idx = user+neighbor if user<neighbor else neighbor+user
                    if edge_idx in edge_feature:
                        edge_attr.append(edge_feature[edge_idx])
                    else:
                        edge_attr.append([0]*12)
    feature_matrix = sorted(feature_matrix)
    feature_matrix = list(map(lambda x: x[1], feature_matrix))

    for agent, fans in community_partition.items():
        community_partition_index[node_projection[agent]] = [node_projection[x] for x in fans if x in node_projection]

    dataset = [feature_matrix, adj_matrix, edge_attr, label_matrix, community_partition_index]

    # save dataset for loading
    np.save(output_path, dataset)
    return dataset


def minmax_data_normalization(source, destination, edge_attr=None):
    '''
    Normalize dataset along each feature dimensions based on corresponding min and max values.
    Input:
        source path
        destination path
    output:
        normalized dataset (np.array)
    '''

    dataset = np.load(source)

    # normalize node features
    feature_matrix = []
    feature_index = [0]
    for u in dataset:
        feature_matrix += u[0]
        feature_index.append(feature_index[-1]+len(u[0]))

    feature_matrix = np.array(feature_matrix)
    feature_matrix = feature_matrix - np.min(feature_matrix, axis=0)
    feature_matrix_max = np.max(feature_matrix, axis=0)
    feature_matrix_max[7:19] = np.max(feature_matrix_max[7:19])
    feature_matrix = feature_matrix/feature_matrix_max
    feature_matrix = feature_matrix.tolist()

    for i,u in enumerate(dataset):
        dataset[i][0] = feature_matrix[feature_index[i]:feature_index[i+1]]

    # Normalize edge features
    if edge_attr:
        edge_feature_matrix = []
        edge_feature_index = [0]
        for u in dataset:
            edge_feature_matrix += u[2]
            edge_feature_index.append(edge_feature_index[-1]+len(u[2]))

        edge_feature_matrix = np.array(edge_feature_matrix)
        edge_feature_matrix = edge_feature_matrix - np.min(edge_feature_matrix)
        edge_feature_matrix = edge_feature_matrix/np.max(edge_feature_matrix)
        edge_feature_matrix = edge_feature_matrix.tolist()

        for i,u in enumerate(dataset):
            dataset[i][2] = edge_feature_matrix[edge_feature_index[i]:edge_feature_index[i+1]]
    
    # save dataset for loading
    np.save(destination, dataset)

    return dataset


def minmax_data_normalization_for_g(source, destination):
    '''
    0:gender是否存在 1:男？ 2:女？ 3:age是否存在 4:age 5:user_type 6:reg_time 7:rlat_create_time 
    8-20:node_history_purchase list 20:node_degree

    Normalize dataset along each feature dimensions based on corresponding min and max values.
    Input:
        source path
        destination path
    output:
        normalized dataset (np.array)
    '''

    dataset = np.load(source)

    feature_matrix = np.array(dataset[0])
    feature_matrix_min = np.min(feature_matrix, axis=0)
    feature_matrix_min[8:20] = np.min(feature_matrix_min[8:20])
    feature_matrix = feature_matrix - feature_matrix_min
    feature_matrix_max = np.max(feature_matrix, axis=0)
    feature_matrix_max[8:20] = np.max(feature_matrix_max[8:20])
    feature_matrix = feature_matrix/feature_matrix_max
    feature_matrix = feature_matrix.tolist()
    dataset[0] = feature_matrix

    edge_features = np.array(dataset[2])
    edge_features_min = np.min(edge_features, axis=0)
    edge_features_min[:] = np.min(edge_features_min)
    edge_features = edge_features - edge_features_min
    edge_features_max = np.max(edge_features, axis=0)
    edge_features_max[:] = np.max(edge_features_max)
    edge_features = edge_features/edge_features_max
    edge_features = edge_features.tolist()
    dataset[2] = edge_features

    # save dataset for loading
    np.save(destination, dataset)

    return dataset


def to_rank(source, destination):
    dataset = np.load('../data/{}.npy'.format(source))
    label = dataset[:,2]
    rank = label.argsort()
    for i,r in enumerate(rank):
        dataset[r,2] = len(label) - i - 1
    # save dataset for loading
    np.save('../data/{}.npy'.format(destination), dataset)


def to_rank_global(source, destination):
    dataset = np.load('../data/{}.npy'.format(source))
    label = dataset[2]
    rank = np.argsort(label)
    res = [0]*len(label)
    for i,r in enumerate(rank):
        res[r] = len(label) - i - 1
    dataset[2] = res
    # save dataset for loading
    np.save('../data/{}.npy'.format(destination), dataset)


def id_embedding(source, destination):
    dataset = np.load(source)
    for i in range(len(dataset)):
        for j in range(len(dataset[i][0])):
            dataset[i][0][j].append(j)
    np.save(destination, dataset)
    return dataset





# dataset = rf_processing()


feature_path = '../data/sample2_features'
community_path = '../data/sample2_community'
label_path = '../data/sample2_label'
edge_feature_path = '../data/sample2_edge_features'
edge_path = '../data/sample2_edges'

output_path = '../data/sample2_dataset.npy'
output_path_unc = '../data/sample2_dataset_unc.npy'

# dataset = global_gcn_processing(feature_path, community_path, label_path, 
#     output_path, edge_feature_path, edge_path, False)

# source = '../data/sample2_dataset.npy'
# destination = '../data/sample2_dataset_norm.npy'
# dataset = minmax_data_normalization_for_g(source, destination)


dataset = global_gcn_processing(feature_path, community_path, label_path, 
    output_path_unc, edge_feature_path, edge_path, True)

source = '../data/sample2_dataset_unc.npy'
destination = '../data/sample2_dataset_unc_norm.npy'
dataset = minmax_data_normalization_for_g(source, destination)

