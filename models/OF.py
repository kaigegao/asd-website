import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class OrthogonalFusion(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, feat1, feat2):
        # feat1 shape [batch, seq_len, d_model]
        # feat2 shape [batch, hidden_dim]
        # calculate feat2 in feat1's space orthogonally projection
        # print("feat1 shape", feat1.size())
        # print("feat2 shape", feat2.size())
        feat2_norm = torch.norm(feat2, p=2, dim=1)
        # print("feat2_norm shape", feat2_norm.size())
        # # 计算feat1和feat2的点积，然后除以两者的范数，得到投影矩阵
        # print("feat2 unsqueeze shape", feat2.unsqueeze(1).size())
        # print("feat1 permute shape", feat1.permute(0, 2, 1).size())
        projection = torch.bmm(feat2.unsqueeze(1), feat1.permute(0, 2, 1))
        projection = torch.bmm(feat2.unsqueeze(2), projection).view(feat1.size())
        projection = projection / (feat2_norm * feat2_norm).view(-1, 1, 1)
        orthogonal_comp = feat1 - projection
        # print("orthogonal_comp shape", orthogonal_comp.size())
        # feat2 = feat2.unsqueeze(-1).permute(0, 2, 1)
        # print("feat2 expand shape", feat2.size())
        # 将feat1维度从[batch,seq_len,enc_in]压缩到[batch,enc_in]
        orthogonal_comp = orthogonal_comp.mean(dim=1)
        out = torch.cat([feat2, orthogonal_comp], dim=-1)
        # print("out shape", out.size())

        # projection = torch.bmm(feat2.unsqueeze(1), torch.flatten(feat1, start_dim=2))
        # projection = torch.bmm(feat2.unsqueeze(2), projection).view(feat1.size())
        # projection = projection / (feat2_norm * feat2_norm).view(-1, 1, 1, 1)
        # orthogonal_comp = feat1 - projection
        # feat2 = feat2.unsqueeze(-1).unsqueeze(-1)
        # out = torch.cat([feat2.expand(orthogonal_comp.size()), orthogonal_comp], dim=1)
        return out
