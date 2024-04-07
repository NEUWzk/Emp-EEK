import torch
import torch.nn as nn
import logging
from gcn_support import EdgeAtt, GCN, batch_graphify


class DialogueGCN(nn.Module):

    def __init__(self, vocab, embedding):  #到这里正确
        super(DialogueGCN, self).__init__()
        g_dim = 200
        h1_dim = 100
        h2_dim = 100
        self.vocab = vocab
        self.glove_embdding = embedding
        self.edge_att = EdgeAtt(g_dim)  #EdgeAtt
        self.gcn = GCN(g_dim, h1_dim, h2_dim)  #调用构造函数，进行初始化
        self.glove_linear = nn.Linear(300, 200)
        self.linear = nn.Linear(300, 100)

        edge_type_to_idx = {'Self': 0, 'TokenToToken': 1, 'TokenToConcept': 2}  #三种边的类型
        self.edge_type_to_idx = edge_type_to_idx
        logging.info(" :", self.edge_type_to_idx)

    def get_rep(self, data):
         # node_features [batch_size,node_num,300]  data['node_id']:(16,18)
        node_features = self.glove_embdding(data["node_id"]) # (16,18,300)-->对data['node_id']进行嵌入
        node_features = self.glove_linear(node_features)  #(16,18,200)
        features, edge_index, edge_norm, edge_type = batch_graphify(
            node_features, data["node_number"], data["node_type"], data["final_dict"], data['node2id'],
            data['keyword2id'], data['concepts_batch'],
            self.edge_type_to_idx, self.edge_att)
        #features(273,200) edge_index(2,970)  edge_norm(970) edge_type(970)
        # edge_norm = None
        graph_out = self.gcn(features, edge_index, edge_norm, edge_type)  #tensor(node2id_num,100)

        return graph_out, features  #（node2id_num,100） （node2id_num,100）

    def forward(self, data):  #data就是前面的batch
        graph_out, features = self.get_rep(data)  #graph_out(node2id_num,100)  features(node2id_num,200)
        out = torch.cat([features, graph_out], dim=-1)  #(node2id_num,300)
        out = self.linear(out)  #(node2id,100)
        return out


