import torch
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import GraphConv
from torch_geometric.nn import APPNP
from torch_geometric.nn import SAGPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_scatter import scatter_max, scatter_mean


import sys
import time

from functions import *
from layers import *
from inits import *


def pretrain_id_embedding(model, data, epochs):
    pretrain_optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()
    start = time.time()
    for epoch in range(epochs):
        pretrain_optimizer.zero_grad()
        loss = model.loss(data.edge_index)
        loss.backward()
        pretrain_optimizer.step()
        time_iter = time.time() - start
        print('Time {:.4f}: Epoch: {}, Loss: {:.4f}'.format(time_iter, epoch, loss))
    torch.save(model.state_dict(),'../model/pretrain_model')


class GCN(torch.nn.Module):
    def __init__(self,args):
        super(GCN, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.n_hidden = args.n_hidden
        self.dropout_ratio = args.dropout_ratio
        self.num_communities = args.num_communities
        self.n_embedding = args.n_embedding
        self.num_nodes = args.num_nodes
        self.device = args.device

        self.embedding1 = torch.nn.Linear(8, self.n_embedding)
        self.embedding2 = torch.nn.Linear(12, self.n_embedding)
        self.embedding3 = torch.nn.Linear(2*self.n_embedding, 2*self.n_embedding)
        
        self.conv1 = GCNConv(2*self.n_embedding, self.n_hidden)
        self.conv2 = GCNConv(self.n_hidden, self.n_hidden)

        self.lin1 = torch.nn.Linear(2*self.n_hidden, self.n_hidden)
        self.lin2 = torch.nn.Linear(self.n_hidden, 1)


    def community_pooling(self, x, community, multi_community_nodes, multi_community_index):
        x = torch.cat((x,x[multi_community_nodes,:]), dim=0)
        community = torch.cat((community.view(-1), multi_community_index), dim=0)
        community = community.view(-1,1).repeat(1,x.size()[1])
        res1 = scatter_mean(x, community, dim=0, dim_size=self.num_communities)
        res2, _ = scatter_max(x, community, dim=0, dim_size=self.num_communities)

        return torch.cat([res1,res2], dim=1)


    def community_pooling2(self, x, community, multi_community_nodes, multi_community_index):
        community = community.view(-1,1).repeat(1, x.size()[1])
        res1 = scatter_mean(x, community, dim=0, dim_size=self.num_communities)
        res2, _ = scatter_max(x, community, dim=0, dim_size=self.num_communities)
        return torch.cat([res1,res2], dim=1)
        

    def forward(self, data):
        x, edge_index, community = data.x, data.edge_index, data.community
        multi_community_nodes, multi_community_index = data.multi_community_nodes, data.multi_community_index

        x1 = F.relu(self.embedding1(x[:,:8]))
        x2 = F.relu(self.embedding2(x[:,8:]))
        x = torch.cat([x1,x2], dim=1)
        x = F.relu(self.embedding3(x))

        x = F.relu(self.conv1(x, edge_index))
        x1 = self.community_pooling2(x, community, multi_community_nodes, multi_community_index)

        x = F.relu(self.conv2(x, edge_index))
        x2 = self.community_pooling2(x, community, multi_community_nodes, multi_community_index)

        x = x1 + x2
        # x = torch.cat([x1,x2],dim=1)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = self.lin2(x).squeeze()

        return x


class GCN_EL(torch.nn.Module):
    def __init__(self,args):
        super(GCN_EL, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.num_edge_features = args.num_edge_features
        self.n_hidden = args.n_hidden
        self.dropout_ratio = args.dropout_ratio
        self.num_communities = args.num_communities
        self.n_embedding = args.n_embedding
        self.num_nodes = args.num_nodes
        self.device = args.device
        self.n_demographic = args.n_demographic

        self.edge_embedding = torch.nn.Linear(self.num_edge_features, self.n_embedding)
        self.embedding1 = torch.nn.Linear(self.n_demographic, self.n_embedding)
        self.embedding2 = torch.nn.Linear(12, self.n_embedding)
        self.embedding3 = torch.nn.Linear(2*self.n_embedding, 2*self.n_embedding)

        self.edge_learning = EdgeLearning(2*self.n_embedding, self.n_embedding, 2*self.n_embedding)
        
        self.conv1 = ELConv(2*self.n_embedding, self.n_hidden, 2*self.n_embedding)
        self.conv2 = ELConv(self.n_hidden, self.n_hidden, 2*self.n_embedding)

        self.lin1 = torch.nn.Linear(2*self.n_hidden, self.n_hidden)
        self.lin2 = torch.nn.Linear(self.n_hidden, 1)


    def community_pooling(self, x, community, multi_community_nodes, multi_community_index):
        x = torch.cat((x,x[multi_community_nodes,:]), dim=0)
        community = torch.cat((community.view(-1), multi_community_index), dim=0)
        community = community.view(-1,1).repeat(1,x.size()[1])
        res1 = scatter_mean(x, community, dim=0, dim_size=self.num_communities)
        res2, _ = scatter_max(x, community, dim=0, dim_size=self.num_communities)

        return torch.cat([res1,res2], dim=1)
        # return res1


    def community_pooling2(self, x, community, multi_community_nodes, multi_community_index):
        community = community.view(-1,1).repeat(1, x.size()[1])
        res1 = scatter_mean(x, community, dim=0, dim_size=self.num_communities)
        res2, _ = scatter_max(x, community, dim=0, dim_size=self.num_communities)
        return torch.cat([res1,res2], dim=1)


    def forward(self, data):
        x, edge_index, edge_attr, community = data.x, data.edge_index, data.edge_attr, data.community
        multi_community_nodes, multi_community_index = data.multi_community_nodes, data.multi_community_index

        edge_attr = F.relu(self.edge_embedding(edge_attr))

        x1 = F.relu(self.embedding1(x[:,:self.n_demographic]))
        x2 = F.relu(self.embedding2(x[:,self.n_demographic:]))
        x = torch.cat([x1,x2], dim=1)
        x = F.relu(self.embedding3(x))

        mask = self.edge_learning(x, edge_index, edge_attr)

        x = F.relu(self.conv1(x, edge_index, mask))
        x1 = self.community_pooling2(x, community, multi_community_nodes, multi_community_index)

        x = F.relu(self.conv2(x, edge_index, mask))
        x2 = self.community_pooling2(x, community, multi_community_nodes, multi_community_index)

        x = x1 + x2
        # x = torch.cat([x1,x2],dim=1)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = self.lin2(x).squeeze()

        return x


class GCN_EL2(torch.nn.Module):
    def __init__(self,args):
        super(GCN_EL2, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.num_edge_features = args.num_edge_features
        self.n_hidden = args.n_hidden
        self.dropout_ratio = args.dropout_ratio
        self.num_communities = args.num_communities
        self.n_embedding = args.n_embedding
        self.num_nodes = args.num_nodes
        self.device = args.device

        self.edge_embedding = torch.nn.Linear(self.num_edge_features, self.n_embedding)
        self.embedding1 = torch.nn.Linear(8, self.n_embedding)
        self.embedding2 = torch.nn.Linear(12, self.n_embedding)
        self.embedding3 = torch.nn.Linear(2*self.n_embedding, 2*self.n_embedding)

        self.edge_learning = EdgeLearning(2*self.n_embedding, self.n_embedding, 2*self.n_embedding)
        
        self.conv1 = ELConv2(2*self.n_embedding, self.n_hidden, 2*self.n_embedding)
        self.conv2 = ELConv2(self.n_hidden, self.n_hidden, 2*self.n_embedding)

        self.lin1 = torch.nn.Linear(2*self.n_hidden, self.n_hidden)
        self.lin2 = torch.nn.Linear(self.n_hidden, 1)


    def community_pooling(self, x, community, multi_community_nodes, multi_community_index):
        x = torch.cat((x,x[multi_community_nodes,:]), dim=0)
        community = torch.cat((community.view(-1), multi_community_index), dim=0)
        community = community.view(-1,1).repeat(1,x.size()[1])
        res1 = scatter_mean(x, community, dim=0, dim_size=self.num_communities)
        res2, _ = scatter_max(x, community, dim=0, dim_size=self.num_communities)

        return torch.cat([res1,res2], dim=1)
        # return res1

    def forward(self, data):
        x, edge_index, edge_attr, community = data.x, data.edge_index, data.edge_attr, data.community
        multi_community_nodes, multi_community_index = data.multi_community_nodes, data.multi_community_index

        edge_attr = F.relu(self.edge_embedding(edge_attr))

        x1 = F.relu(self.embedding1(x[:,:8]))
        x2 = F.relu(self.embedding2(x[:,8:]))
        x = torch.cat([x1,x2], dim=1)
        x = F.relu(self.embedding3(x))

        mask = self.edge_learning(x, edge_index, edge_attr)

        x = F.relu(self.conv1(x, edge_index, mask))
        x1 = self.community_pooling(x, community, multi_community_nodes, multi_community_index)

        x = F.relu(self.conv2(x, edge_index, mask))
        x2 = self.community_pooling(x, community, multi_community_nodes, multi_community_index)

        x = x1 + x2
        # x = torch.cat([x1,x2],dim=1)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = self.lin2(x).squeeze()

        return x


class GCN_EL_CP(torch.nn.Module):
    def __init__(self,args):
        super(GCN_EL_CP, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.num_edge_features = args.num_edge_features
        self.n_hidden = args.n_hidden
        self.dropout_ratio = args.dropout_ratio
        self.num_communities = args.num_communities
        self.n_embedding = args.n_embedding
        self.num_nodes = args.num_nodes
        self.device = args.device
        self.attn_weight = Parameter(torch.Tensor(2*self.n_hidden, 1))

        self.edge_embedding = torch.nn.Linear(self.num_edge_features, self.n_embedding)
        self.embedding1 = torch.nn.Linear(8, self.n_embedding)
        self.embedding2 = torch.nn.Linear(12, self.n_embedding)
        self.embedding3 = torch.nn.Linear(2*self.n_embedding, 2*self.n_embedding)

        self.edge_learning = EdgeLearning(2*self.n_embedding, self.n_embedding, 2*self.n_embedding)
        
        self.conv1 = ELConv(2*self.n_embedding, self.n_hidden, 2*self.n_embedding)
        self.conv2 = ELConv(self.n_hidden, self.n_hidden, 2*self.n_embedding)

        self.attn_mlp = torch.nn.Linear(2*self.n_hidden, 2*self.n_hidden)

        self.lin1 = torch.nn.Linear(4*self.n_hidden, 2*self.n_hidden)
        self.lin2 = torch.nn.Linear(2*self.n_hidden, 1)

        self.reset_parameters()


    def reset_parameters(self):
        glorot(self.attn_weight)


    def community_pooling(self, x, community_for_pooling, pooling_edges):
        """
        attention based
        """
        community = community_for_pooling.view(-1,1).repeat(1,x.size()[1]*2)

        x_i = x[pooling_edges[0,:]]
        x_j = x[pooling_edges[1,:]]
        x_pooling = torch.cat((x_i, x_j), dim=-1)
        x_pooling = F.relu(self.attn_mlp(x_pooling))

        alpha = torch.matmul(x_pooling, self.attn_weight)
        alpha = scatter_softmax(alpha, community_for_pooling.view(-1,1), dim=0)

        x_pooling = x_pooling*alpha

        res1 = scatter_mean(x_pooling, community, dim=0, dim_size=self.num_communities)
        res2, _ = scatter_max(x_pooling, community, dim=0, dim_size=self.num_communities)

        return torch.cat([res1,res2], dim=1)
        # return res1

    def community_pooling2(self, x, community_for_pooling, pooling_edges):
        x_i = x[pooling_edges[0,:]]
        x_j = x[pooling_edges[1,:]]
        x_pooling = torch.cat((x_i, x_j), dim=-1)
        x_pooling = F.relu(self.attn_mlp(x_pooling))

        community_for_pooling = community_for_pooling.view(-1,1).repeat(1,x.size()[1]*2)

        res1 = scatter_mean(x_pooling, community_for_pooling, dim=0, dim_size=self.num_communities)
        res2, _ = scatter_max(x_pooling, community_for_pooling, dim=0, dim_size=self.num_communities)

        return torch.cat([res1, res1], dim=1)


    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        community_idx_for_pooling, pooling_edges = data.community_idx_for_pooling, data.pooling_edges
        community = data.community

        edge_attr = F.relu(self.edge_embedding(edge_attr))

        x1 = F.relu(self.embedding1(x[:,:8]))
        x2 = F.relu(self.embedding2(x[:,8:]))
        x = torch.cat([x1,x2], dim=1)
        x = F.relu(self.embedding3(x))

        mask = self.edge_learning(x, edge_index, edge_attr)

        x = F.relu(self.conv1(x, edge_index, mask))
        # x1 = self.community_pooling(x, community_idx_for_pooling, pooling_edges)
        x1 = self.community_pooling2(x, community_idx_for_pooling, pooling_edges)

        x = F.relu(self.conv2(x, edge_index, mask))
        # x2 = self.community_pooling(x, community_idx_for_pooling, pooling_edges)
        x2 = self.community_pooling2(x, community_idx_for_pooling, pooling_edges)

        x = x1 + x2
        # x = torch.cat([x1,x2],dim=1)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = self.lin2(x).squeeze()

        return x



class GCN_EL_H(torch.nn.Module):
    def __init__(self,args):
        super(GCN_EL_H, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.num_edge_features = args.num_edge_features
        self.n_hidden = args.n_hidden
        self.dropout_ratio = args.dropout_ratio
        self.num_communities = args.num_communities
        self.n_embedding = args.n_embedding
        self.num_nodes = args.num_nodes
        self.device = args.device

        self.edge_embedding1 = torch.nn.Linear(self.num_edge_features, self.n_embedding)
        self.edge_embedding2 = torch.nn.Linear(self.num_edge_features, self.n_embedding)

        self.embedding1 = torch.nn.Linear(8, self.n_embedding)
        self.embedding2 = torch.nn.Linear(12, self.n_embedding)
        self.embedding3 = torch.nn.Linear(2*self.n_embedding, 2*self.n_embedding)

        self.edge_learning1 = EdgeLearning(2*self.n_embedding, self.n_embedding, 2*self.n_embedding)
        self.edge_learning2 = EdgeLearning(2*self.n_embedding, self.n_embedding, 2*self.n_embedding)
                
        self.conv_inter1 = ELConv(2*self.n_embedding, self.n_hidden, 2*self.n_embedding)
        self.conv_intra1 = ELConv(2*self.n_embedding, self.n_hidden, 2*self.n_embedding)

        self.conv_inter2 = ELConv(self.n_hidden, self.n_hidden, 2*self.n_embedding)
        self.conv_intra2 = ELConv(self.n_hidden, self.n_hidden, 2*self.n_embedding)

        self.lin1 = torch.nn.Linear(2*self.n_hidden, self.n_hidden)
        self.lin2 = torch.nn.Linear(self.n_hidden, 1)


    def community_pooling(self, x, community, multi_community_nodes, multi_community_index):
        x = torch.cat((x,x[multi_community_nodes,:]), dim=0)
        community = torch.cat((community.view(-1), multi_community_index), dim=0)
        community = community.view(-1,1).repeat(1,x.size()[1])
        res1 = scatter_mean(x, community, dim=0, dim_size=self.num_communities)
        res2, _ = scatter_max(x, community, dim=0, dim_size=self.num_communities)

        return torch.cat([res1,res2], dim=1)
        # return res1


    def community_pooling2(self, x, community, multi_community_nodes, multi_community_index):
        community = community.view(-1,1).repeat(1, x.size()[1])
        res1 = scatter_mean(x, community, dim=0, dim_size=self.num_communities)
        res2, _ = scatter_max(x, community, dim=0, dim_size=self.num_communities)
        return torch.cat([res1,res2], dim=1)


    def forward(self, data):
        x, edge_index, edge_attr, community = data.x, data.edge_index, data.edge_attr, data.community
        multi_community_nodes, multi_community_index = data.multi_community_nodes, data.multi_community_index
        adj_inter, adj_intra = data.adj_inter, data.adj_intra
        edge_attr_inter, edge_attr_intra = data.edge_attr_inter, data.edge_attr_intra

        edge_attr_inter = F.relu(self.edge_embedding1(edge_attr_inter))
        edge_attr_intra = F.relu(self.edge_embedding2(edge_attr_intra))

        x1 = F.relu(self.embedding1(x[:,:8]))
        x2 = F.relu(self.embedding2(x[:,8:]))
        x = torch.cat([x1,x2], dim=1)
        x = F.relu(self.embedding3(x))

        mask_inter = self.edge_learning1(x, adj_inter, edge_attr_inter)
        mask_intra = self.edge_learning2(x, adj_intra, edge_attr_intra)

        x_intra = x.clone()
        x = F.relu(self.conv_inter1(x, adj_inter, mask_inter))
        x += F.relu(self.conv_intra1(x_intra, adj_intra, mask_intra))
        x1 = self.community_pooling(x, community, multi_community_nodes, multi_community_index)

        x_intra = x.clone()
        x = F.relu(self.conv_inter2(x, adj_inter, mask_inter))
        x += F.relu(self.conv_intra2(x_intra, adj_intra, mask_intra))
        x2 = self.community_pooling(x, community, multi_community_nodes, multi_community_index)

        x = x1 + x2
        # x = torch.cat([x1,x2],dim=1)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = self.lin2(x).squeeze()

        return x



class GCN_EL_CP_H(torch.nn.Module):
    def __init__(self,args):
        super(GCN_EL_CP_H, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.num_edge_features = args.num_edge_features
        self.n_hidden = args.n_hidden
        self.dropout_ratio = args.dropout_ratio
        self.num_communities = args.num_communities
        self.n_embedding = args.n_embedding
        self.num_nodes = args.num_nodes
        self.device = args.device
        self.attn_weight = Parameter(torch.Tensor(2*self.n_hidden, 1))

        self.edge_embedding = torch.nn.Linear(self.num_edge_features, self.n_embedding)
        self.embedding1 = torch.nn.Linear(8, self.n_embedding)
        self.embedding2 = torch.nn.Linear(12, self.n_embedding)
        self.embedding3 = torch.nn.Linear(2*self.n_embedding, 2*self.n_embedding)

        self.edge_learning = EdgeLearning(2*self.n_embedding, self.n_embedding, 2*self.n_embedding)
        
        self.conv1 = ELConv(2*self.n_embedding, self.n_hidden, 2*self.n_embedding)
        self.conv2 = ELConv(self.n_hidden, self.n_hidden, 2*self.n_embedding)

        self.attn_mlp = torch.nn.Linear(2*self.n_hidden, 2*self.n_hidden)

        self.lin1 = torch.nn.Linear(4*self.n_hidden, 2*self.n_hidden)
        self.lin2 = torch.nn.Linear(2*self.n_hidden, 1)

        self.reset_parameters()


    def reset_parameters(self):
        glorot(self.attn_weight)


    def community_pooling(self, x, community_for_pooling, pooling_edges):
        """
        attention based
        """
        community = community_for_pooling.view(-1,1).repeat(1,x.size()[1]*2)

        x_i = x[pooling_edges[0,:]]
        x_j = x[pooling_edges[1,:]]
        x_pooling = torch.cat((x_i, x_j), dim=-1)
        x_pooling = F.relu(self.attn_mlp(x_pooling))

        alpha = torch.matmul(x_pooling, self.attn_weight)
        alpha = scatter_softmax(alpha, community_for_pooling.view(-1,1), dim=0)

        x_pooling = x_pooling*alpha

        res1 = scatter_mean(x_pooling, community, dim=0, dim_size=self.num_communities)
        res2, _ = scatter_max(x_pooling, community, dim=0, dim_size=self.num_communities)

        return torch.cat([res1,res2], dim=1)
        # return res1

    def community_pooling2(self, x, community_for_pooling, pooling_edges):
        x_i = x[pooling_edges[0,:]]
        x_j = x[pooling_edges[1,:]]
        x_pooling = torch.cat((x_i, x_j), dim=-1)
        x_pooling = F.relu(self.attn_mlp(x_pooling))

        community_for_pooling = community_for_pooling.view(-1,1).repeat(1,x.size()[1]*2)

        res1 = scatter_mean(x_pooling, community_for_pooling, dim=0, dim_size=self.num_communities)
        res2, _ = scatter_max(x_pooling, community_for_pooling, dim=0, dim_size=self.num_communities)

        return torch.cat([res1, res1], dim=1)


    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        community_idx_for_pooling, pooling_edges = data.community_idx_for_pooling, data.pooling_edges
        community = data.community

        edge_attr = F.relu(self.edge_embedding(edge_attr))

        x1 = F.relu(self.embedding1(x[:,:8]))
        x2 = F.relu(self.embedding2(x[:,8:]))
        x = torch.cat([x1,x2], dim=1)
        x = F.relu(self.embedding3(x))

        mask = self.edge_learning(x, edge_index, edge_attr)

        x = F.relu(self.conv1(x, edge_index, mask))
        # x1 = self.community_pooling(x, community_idx_for_pooling, pooling_edges)
        x1 = self.community_pooling2(x, community_idx_for_pooling, pooling_edges)

        x = F.relu(self.conv2(x, edge_index, mask))
        # x2 = self.community_pooling(x, community_idx_for_pooling, pooling_edges)
        x2 = self.community_pooling2(x, community_idx_for_pooling, pooling_edges)

        x = x1 + x2
        # x = torch.cat([x1,x2],dim=1)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = self.lin2(x).squeeze()

        return x










class GCN_ID(torch.nn.Module):
    def __init__(self,args,id_embeddings):
        super(GCN_ID, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.n_hidden = args.n_hidden
        self.dropout_ratio = args.dropout_ratio
        self.num_communities = args.num_communities
        self.n_id_embedding = args.n_id_embedding
        self.n_embedding = args.n_embedding
        self.num_nodes = args.num_nodes
        self.device = args.device
        self.id_embeddings = id_embeddings.to(self.args.device)

        embedding1 = torch.nn.Linear(7, self.n_embedding)
        embedding2 = torch.nn.Linear(12, self.n_embedding)
        embedding3 = torch.nn.Linear(2*self.n_embedding   +self.n_id_embedding, 2*self.n_embedding+self.n_id_embedding)
        
        conv1 = GCNConv(2*self.n_embedding+self.n_id_embedding, self.n_hidden)
        conv2 = GCNConv(self.n_hidden, self.n_hidden)

        lin1 = torch.nn.Linear(self.n_hidden*2, self.n_hidden)
        lin2 = torch.nn.Linear(self.n_hidden, 1)

        self.net = torch.nn.ModuleList([embedding1,embedding2,embedding3,conv1,conv2,lin1,lin2])


    def community_pooling(self, x, community):
        community = community.view(-1,1).repeat(1,x.size()[1])
        res1 = scatter_mean(x, community, dim=0, dim_size=self.num_communities)
        # res2, _ = my_scatter_max(x, community, dim=0, dim_size=self.num_communities)

        return torch.cat([res1,res1], dim=1)
        

    def forward(self, data):
        x, edge_index, community = data.x, data.edge_index, data.community

        x1 = F.relu(self.net[0](x[:,:7]))
        x2 = F.relu(self.net[1](x[:,7:]))
        index = torch.tensor(range(self.num_nodes)).to(self.device)

        x = torch.cat([x1,x2,self.id_embeddings], dim=1)
        x = F.relu(self.net[2](x))

        x = F.relu(self.net[3](x, edge_index))
        x1 = self.community_pooling(x, community)

        x = F.relu(self.net[4](x, edge_index))
        x2 = self.community_pooling(x, community)

        x = x1 + x2
        # x = torch.cat([x1,x2] ,dim=1)

        x = F.relu(self.net[5](x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = self.net[6](x).squeeze()

        return x


class GCN_ID_bak(torch.nn.Module):
    def __init__(self,args,id_embeddings):
        super(GCN_ID, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.n_hidden = args.n_hidden
        self.dropout_ratio = args.dropout_ratio
        self.num_communities = args.num_communities
        self.n_id_embedding = args.n_id_embedding
        self.n_embedding = args.n_embedding
        self.num_nodes = args.num_nodes
        self.device = args.device
        self.id_embeddings = id_embeddings

        self.embedding1 = torch.nn.Linear(7, self.n_embedding)
        self.embedding2 = torch.nn.Linear(12, self.n_embedding)
        self.embedding3 = torch.nn.Linear(2*self.n_embedding+self.n_id_embedding, 2*self.n_embedding+self.n_id_embedding)
        
        self.conv1 = GCNConv(2*self.n_embedding+self.n_id_embedding, self.n_hidden)
        self.conv2 = GCNConv(self.n_hidden, self.n_hidden)

        self.lin1 = torch.nn.Linear(self.n_hidden*2, self.n_hidden)
        self.lin2 = torch.nn.Linear(self.n_hidden, 1)


    def community_pooling(self, x, community):
        community = community.view(-1,1).repeat(1,x.size()[1])
        res1 = scatter_mean(x, community, dim=0, dim_size=self.num_communities)
        # res2, _ = my_scatter_max(x, community, dim=0, dim_size=self.num_communities)

        return torch.cat([res1,res1], dim=1)

    def forward(self, data):
        x, edge_index, community = data.x, data.edge_index, data.community

        x1 = F.relu(self.embedding1(x[:,:7]))
        x2 = F.relu(self.embedding2(x[:,7:]))
        index = torch.tensor(range(self.num_nodes)).to(self.device)

        x = torch.cat([x1,x2,self.id_embeddings], dim=1)
        x = F.relu(self.embedding3(x))

        x = F.relu(self.conv1(x, edge_index))
        x1 = self.community_pooling(x, community)

        x = F.relu(self.conv2(x, edge_index))
        x2 = self.community_pooling(x, community)

        x = x1 + x2
        # x = torch.cat([x1,x2] ,dim=1)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = self.lin2(x).squeeze()

        return x



class GAT(torch.nn.Module):
    def __init__(self,args):
        super(GAT, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.n_hidden = args.n_hidden
        self.dropout_ratio = args.dropout_ratio
        
        self.conv1 = GATConv(self.num_features, self.n_hidden, heads=1)
        self.conv2 = GATConv(self.n_hidden, self.n_hidden, heads=1)

        self.lin1 = torch.nn.Linear(self.n_hidden*2, self.n_hidden)
        self.lin2 = torch.nn.Linear(self.n_hidden, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.conv1(x, edge_index))
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index))
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1 + x2

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = self.lin2(x).squeeze()

        return x


class SAGE(torch.nn.Module):
    def __init__(self,args):
        super(SAGE, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.n_hidden = args.n_hidden
        self.dropout_ratio = args.dropout_ratio
        
        self.conv1 = SAGEConv(self.num_features, self.n_hidden)
        self.conv2 = SAGEConv(self.n_hidden, self.n_hidden)

        self.lin1 = torch.nn.Linear(self.n_hidden*2, self.n_hidden)
        self.lin2 = torch.nn.Linear(self.n_hidden, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.conv1(x, edge_index))
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index))
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1 + x2

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = self.lin2(x).squeeze()

        return x


class SAGP(torch.nn.Module):
    def __init__(self,args):
        super(SAGP, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.n_hidden = args.n_hidden
        self.pooling_ratio = args.pooling_ratio
        self.dropout_ratio = args.dropout_ratio
        
        self.conv1 = GCNConv(self.num_features, self.n_hidden)
        self.pool1 = SAGPooling(self.n_hidden, ratio=self.pooling_ratio)
        self.conv2 = GCNConv(self.n_hidden, self.n_hidden)
        self.pool2 = SAGPooling(self.n_hidden, ratio=self.pooling_ratio)

        self.lin1 = torch.nn.Linear(self.n_hidden*2, self.n_hidden)
        self.lin2 = torch.nn.Linear(self.n_hidden, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1 + x2

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = self.lin2(x).squeeze()

        return x

class SAGP(torch.nn.Module):
    def __init__(self,args):
        super(SAGP, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.n_hidden = args.n_hidden
        self.pooling_ratio = args.pooling_ratio
        self.dropout_ratio = args.dropout_ratio
        
        self.conv1 = GCNConv(self.num_features, self.n_hidden)
        self.pool1 = SAGPooling(self.n_hidden, ratio=self.pooling_ratio)
        self.conv2 = GCNConv(self.n_hidden, self.n_hidden)
        self.pool2 = SAGPooling(self.n_hidden, ratio=self.pooling_ratio)

        self.lin1 = torch.nn.Linear(self.n_hidden*2, self.n_hidden)
        self.lin2 = torch.nn.Linear(self.n_hidden, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1 + x2

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = self.lin2(x).squeeze()

        return x

# class GCNSAGP(torch.nn.Module):
#     def __init__(self,args):
#         super(Net, self).__init__()
#         self.args = args
#         self.num_features = args.num_features
#         self.n_hidden = args.n_hidden
#         self.num_classes = args.num_classes
#         self.pooling_ratio = args.pooling_ratio
#         self.dropout_ratio = args.dropout_ratio
        
#         self.conv1 = GCNConv(self.num_features, self.n_hidden)
#         self.pool1 = SAGPool(self.n_hidden, ratio=self.pooling_ratio)
#         self.conv2 = GCNConv(self.n_hidden, self.n_hidden)
#         self.pool2 = SAGPool(self.n_hidden, ratio=self.pooling_ratio)
#         self.conv3 = GCNConv(self.n_hidden, self.n_hidden)
#         self.pool3 = SAGPool(self.n_hidden, ratio=self.pooling_ratio)

#         self.lin1 = torch.nn.Linear(self.n_hidden*2, self.n_hidden)
#         self.lin2 = torch.nn.Linear(self.n_hidden, self.n_hidden//2)
#         self.lin3 = torch.nn.Linear(self.n_hidden//2, self. num_classes)

#     def forward(self, data):
#         x, edge_index, batch = data.x, data.edge_index, data.batch

#         x = F.relu(self.conv1(x, edge_index))
#         x, edge_index, _, batch, _ = self.pool1(x, edge_index, None, batch)
#         x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

#         x = F.relu(self.conv2(x, edge_index))
#         x, edge_index, _, batch, _ = self.pool2(x, edge_index, None, batch)
#         x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

#         x = F.relu(self.conv3(x, edge_index))
#         x, edge_index, _, batch, _ = self.pool3(x, edge_index, None, batch)
#         x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

#         x = x1 + x2 + x3

#         x = F.relu(self.lin1(x))
#         x = F.dropout(x, p=self.dropout_ratio, training=self.training)
#         x = F.relu(self.lin2(x))
#         x = F.log_softmax(self.lin3(x), dim=-1)

#         return x
#     