import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_geometric.nn import global_mean_pool

class GCN_Net(torch.nn.Module):
    def __init__(self, feature, hidden1, hidden2):  #两层GCN
        super(GCN_Net, self).__init__()
        self.conv1 = GCNConv(feature, hidden1)
        self.conv2 = GCNConv(hidden1, hidden2)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch   #载入数据
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)
        return x