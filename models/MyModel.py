import torch
import torch.nn as nn
import torch.nn.functional as F
from models.mamba import TimeSeriesModel
from models.GCN import GCN_Net
from models.OF import OrthogonalFusion
from nilearn import connectome
from torch_geometric.data import Data
from torch_geometric.data.batch import Batch

class MyModel(nn.Module):
    def __init__(self, configs):
        super(MyModel, self).__init__()
        self.configs = configs
        self.timeseries_feature_extractor = TimeSeriesModel(configs, configs.d_model, configs.n_layer, configs.ssm_cfg,
                                                            configs.norm_epsilon, configs.rms_norm,
                                                            configs.initializer_cfg, configs.fused_add_norm,
                                                            configs.residual_in_fp32, configs.device, configs.dtype)
        self.gcn = GCN_Net(feature=configs.seq_len, hidden1=configs.hidden_1, hidden2=configs.hidden_2)
        self.fusion = OrthogonalFusion()
        self.linear = nn.Linear(configs.d_model + configs.hidden_2, configs.num_class)

    def forward(self, X):
        ts_rep = self.timeseries_feature_extractor(X)
        # from timeseries to flatten embeddings and to graph
        conn_measure = connectome.ConnectivityMeasure(kind='correlation')
        all_graphs = []
        for sub in range(X.shape[0]):
            timeseries = X[sub, :].detach().cpu().numpy()
            connctivity = conn_measure.fit_transform([timeseries])[0]
            num_nodes = connctivity.shape[0]
            node_index = torch.arange(num_nodes).view(-1,1)
            edge_index = torch.tensor(connctivity.nonzero(), dtype=torch.float64)
            graph = Data(x=torch.tensor(timeseries.T, dtype=torch.float), edge_index=torch.tensor(edge_index.contiguous(), dtype=torch.long), y=node_index)
            all_graphs.append(graph)
        list_G = list(all_graphs)
        G = Batch.from_data_list(list_G).to(self.configs.device)
        gcn_rep = self.gcn(G)
        rep = self.fusion(ts_rep, gcn_rep)
        out = self.linear(rep)
        del G
        return F.log_softmax(out, dim=1), out, ts_rep, gcn_rep, graph
