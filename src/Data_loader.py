import numpy as np


class Data_loader:
    def __init__(self):
        self.path1 = '/data/mas/yuanyuan/community_value_prediction/sample2_dataset_rf.npy'
        self.path2 = '/data/mas/yuanyuan/community_value_prediction/sample2_dataset_unc_norm.npy'
        self.path_embedding = '/data/mas/yuanyuan/community_value_prediction/node.embeddings'

    def laod(self):
        data = np.load(self.path1,allow_pickle=True)
        feature_matrix, adj_matrix, edge_attr, label_matrix, community_partition_index = np.load(self.path2,allow_pickle=True)
        X = data[:,:-1]
        y = data[:,-1]
        return X,y,feature_matrix, adj_matrix, edge_attr, label_matrix, community_partition_index

    def embed_process(self):
        fp = open(self.path_embedding)
        content = fp.readlines()
        data = content[1:]
        for index,x in enumerate(data):
            data[index] = x[:-1].split(' ')
            data[index] = list(map(float,data[index]))
            data[index] = [data[index][0],data[index][1:]]
        return data
    