import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import RGCNConv, GraphConv

from config import args

class EdgeAtt(nn.Module):

    def __init__(self, g_dim):
        super(EdgeAtt, self).__init__()
        self.wp = 6
        self.wf = 6

        self.weight = nn.Parameter(torch.zeros((g_dim, g_dim)).float(), requires_grad=True)
        var = 2. / (self.weight.size(0) + self.weight.size(1))
        self.weight.data.normal_(0, var)

    def forward(self, node_features, node_num_tensor):
        batch_size, mx_len = node_features.size(0), node_features.size(1)
        alphas = []

        weight = self.weight.unsqueeze(0).unsqueeze(0)  # [1,1,D_g=300,D_g]
        # node_features.unsqueeze [batch,mx,D_g,1]
        att_matrix = torch.matmul(weight, node_features.unsqueeze(-1)).squeeze(-1)  # [B, L, D_g]
        for i in range(batch_size):
            cur_len = node_num_tensor[i].item()
            alpha = torch.zeros((mx_len, mx_len)).cuda()
            for j in range(cur_len):
                s = j - self.wp if j - self.wp >= 0 else 0
                e = j + self.wf if j + self.wf <= cur_len - 1 else cur_len - 1
                tmp = att_matrix[i, s: e + 1, :]  # [L', D_g]
                feat = node_features[i, j]  # [D_g]
                score = torch.matmul(tmp, feat) / pow(200, 0.5)
                probs = F.softmax(score, dim=0)  # [L']
                alpha[j, s: e + 1] = probs
            alphas.append(alpha)

        return alphas

class GCN(nn.Module):

    def __init__(self, g_dim, h1_dim, h2_dim):
        super(GCN, self).__init__()
        self.num_relations = 4
        self.conv1 = RGCNConv(g_dim, h1_dim, self.num_relations, num_bases=30) #输入通道数字，隐藏通道，关系个数，正则化
        self.conv2 = GraphConv(h1_dim, h2_dim)

    def forward(self, node_features, edge_index, edge_norm, edge_type):
        # x = self.conv1(node_features, edge_index, edge_type, edge_norm)  #(这里出错了,应该传递三个参数，但是传了四个)
        x = self.conv1(node_features, edge_index, edge_type)
        x = self.conv2(x, edge_index)

        return x




def batch_graphify(features, lengths, node_type_tensor, final_node_dict, node2id, keyword2id, concepts_batch, edge_type_to_idx, att_model):
    node_features, edge_index, edge_norm, edge_type = [], [], [], []
    batch_size = features.size(0)  #16
    length_sum = 0
    edge_weights = att_model(features, lengths)  #边的初始化权重 list16 可能有问题

    for j in range(batch_size):
        cur_len = lengths[j].item()  # 获取当前输入句子中node_number的数量 14
        node_features.append(features[j, :cur_len, :])  #list[tensor(node_len)]
        edge_ind = edge_perms(cur_len, final_node_dict[j], node2id[j], keyword2id[j], concepts_batch[j])   #node2id[j]: dict{} 长度和lengths对应
        edge_ind_rec = [(item[0] + length_sum, item[1] + length_sum) for item in edge_ind]  # item是一个tuple
        length_sum += cur_len  # 加上偏移量 也就是当前输入的node2id中node数量

        for item, item_rec in zip(edge_ind, edge_ind_rec):
            edge_index.append(torch.tensor([item_rec[0], item_rec[1]]))
            edge_norm.append(edge_weights[j][item[0], item[1]])
            node1 = node_type_tensor[j, item[0]].item()  # 获取当前节点类型（是否关键concept词汇）,关键词为0
            node2 = node_type_tensor[j, item[1]].item()
            node_result = node1 + node2

            if node_result == 0:  #两个都是关键词结点
                edge_type.append(edge_type_to_idx['TokenToToken'])
            elif node_result == 1: #一个关键词结点，一个外部知识
                edge_type.append(edge_type_to_idx['TokenToConcept'])  #有一个关键词

    node_features = torch.cat(node_features, dim=0).cuda()  # 将node2id所有个数加在一起 tensor(256,200)
    edge_index = torch.stack(edge_index).t().contiguous().cuda()  # [2, E] 和两面两个维度必须一致
    edge_norm = torch.stack(edge_norm).cuda()  # ？？？有点问题
    edge_type = torch.tensor(edge_type).long().cuda()  # 标识batch内所有边的类型 1 or 2

    return node_features, edge_index, edge_norm, edge_type




def edge_perms(length, dict, node2id, keyword2id, concepts_batch):
    """
    Method to construct the edges of a graph (a utterance) considering the past and future window.
    return: list of tuples. tuple -> (vertice(int), neighbor(int))
    """
    #length:当前传进来的列表里面有多少node
    all_perms = set()  # 集合类型，有4个关键词，那么集合中就有4*4=16个元素  (0,0) (0,6) (0,7) (0,13) (6,6)...

    #先给所有的关键词token构建边关系
    keyword_list = list(keyword2id.values())
    keyword_len = len(keyword_list)
    if keyword_len > 1:
        for p in range(keyword_len):  # 遍历每句话中的每个关键词
            for q in range(p + 1, keyword_len):  # 再遍历一遍关键词集合,构造关键词-关键词 的关系（边）
                all_perms.add((keyword_list[p], keyword_list[q]))
    else:
        all_perms.add((keyword_list[0],keyword_list[0]))  #如果dict = {}并且关键词只有一个，那么这种情况边会是空的
    perms = set()

    #接下来构建关键词和外部知识之间的边：
    for i in dict.keys():
        value_index = dict[i]  #[sicialize]
        external_lis = [node2id[item] for item in value_index]
        for each in external_lis:  # 1
            perms.add((node2id[i],each))

    all_perms = all_perms.union(perms)  # 合并两个集合 16 && 28 -> 38个
    return list(all_perms)

