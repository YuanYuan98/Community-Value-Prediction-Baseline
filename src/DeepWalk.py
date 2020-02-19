import torch
import torch.nn as nn
import torch.nn.functional as F
from Data_loader import Data_loader
from config import Config
from dataset_processing import create_dataset_global
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_scatter import scatter_max, scatter_mean

class DeepWalk(nn.Module):
    def __init__(self,args):
        super(DeepWalk, self).__init__()
        path = '/home/yuanyuan/workplace/influence/data/sample2_dataset_unc_norm.npy'
        self.config = Config
        self.dataset = create_dataset_global(path)
        self.num_communities = args.num_communities
        self.args = args
        '''
        Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, community=community,
                   pooling_edges=pooling_edges, community_idx_for_pooling=community_idx_for_pooling,
                   multi_community_nodes=multi_community_nodes, multi_community_index=multi_community_index,
                   adj_inter=adj_inter, adj_intra=adj_intra, edge_attr_inter=edge_attr_inter,
                   edge_attr_intra=edge_attr_intra)
        '''
        self.linear_feature = nn.Linear(60,20)
        self.linear_DW = nn.Linear(20,20)
        self.linear = nn.Linear(40,self.config.output_size)
        self.demographic_linear = nn.Linear(8,20)
        self.purchase_linear = nn.Linear(12,20)
    
    def community_pooling(self, x, community, multi_community_nodes, multi_community_index):
        x = torch.cat((x,x[multi_community_nodes,:]), dim=0)
        community = torch.cat((community.view(-1), multi_community_index), dim=0)
        community = community.view(-1,1).repeat(1,x.size()[1])
        res1 = scatter_mean(x, community, dim=0, dim_size=self.num_communities)
        res2, _ = scatter_max(x, community, dim=0, dim_size=self.num_communities)
        return torch.cat([res1,res2], dim=1)
    
    def forward(self,x):

        demographic = F.relu(self.demographic_linear(self.dataset.x[:,:8].to(self.args.device)))
        purchase = F.relu(self.purchase_linear(self.dataset.x[:,8:].to(self.args.device)))

        y = torch.cat((demographic,purchase,x),1)

        y = F.relu(self.linear_feature(y))

        y = self.community_pooling(y,self.dataset.community.to(self.args.device),self.dataset.multi_community_nodes.to(self.args.device),self.dataset.multi_community_index.to(self.args.device))
        # batch_size * embedding_size

        y = F.relu(self.linear(y))

        return y












    


        
