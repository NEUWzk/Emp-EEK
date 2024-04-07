import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from config import args
import config
from GRU_Network import GRU


def init_weights_normal(m):
    if type(m) == nn.Linear:
        torch.nn.init.normal_(m.weight, std=0.02)

class RelationNetwork(nn.Module):

    def __init__(self, embedding):
        super(RelationNetwork, self).__init__()

        self.rel_number = 44  #路径中关系的种类数量
        self.rel_dim = 100
        self.input_dim = 600
        self.out_dim = 300
        self.one_tri_dim = 100
        self.mean_trans = nn.Linear(100,300)  #对两个Node对应的100维列向量求平均，得到300维向量

        self.relation_embedding = nn.Embedding(self.rel_number, self.rel_dim)
        init.xavier_uniform_(self.relation_embedding.weight)
        self.embedding = embedding
        self.path_encoder = GRU(args.emb_dim, args.hidden_dim, bidirectional = True)
        self.output_trans = nn.Sequential(
            nn.Linear(self.input_dim, self.out_dim),
            nn.Tanh(),
        )
        self.output_trans.apply(init_weights_normal)
        self.dim_trans = nn.Sequential(
            nn.Linear(self.input_dim, self.out_dim)
        )
        self.one_tri_trans = nn.Linear(self.one_tri_dim, self.out_dim)
        self.trans_node_to_decoder_dim = nn.Linear(self.one_tri_dim, self.out_dim)
        self.dropout = nn.Dropout(0.1)
        self.layernorm = nn.LayerNorm(self.out_dim)

    def merge_triple(self, sequences):
        dim = sequences[0].size(-1)   #300
        lengths = [seq.size(0) for seq in sequences]  #[全是1]
        padded_seqs = self.embedding(torch.tensor(config.PAD_idx).cuda()).unsqueeze(0).expand(len(sequences), max(lengths), dim)   #tensor(sequence_len,max_seq_len,300)
        padded_seqs_ = padded_seqs.clone()
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs_[i, :end, :] = seq

        return padded_seqs_, lengths


    def prepare_gru_input(self, node, triple):  #node:(cur_node2id_num,100)     triple:[(1,2),(3,4),(5,6)]
        paths = []
        for tri in triple:  #tri:(a,b)
            tri_a = node[tri[0], :].view(1,100)  #tensor(100)
            tri_b = node[tri[-1], :].view(1,100)
            concat = torch.cat((tri_a, tri_b), dim=0)  #tensor(2,100)
            tri_mean = torch.mean(concat, dim=0)
            paths.append(self.mean_trans(tri_mean).view(1,300))

        dia_tri, dia_tri_lengths = self.merge_triple(paths)
        return dia_tri, dia_tri_lengths





    def merge_node(self, sequences):
        dim = sequences[0].size(-1)
        lengths = [seq.size(0) for seq in sequences]
        # padded_seqs = torch.ones(len(sequences), max(lengths), dim).long()  ## padding index -1
        padded_seqs = self.embedding(torch.tensor(config.PAD_idx).cuda()).unsqueeze(0).expand(len(sequences), max(lengths), dim)
        padded_seqs_ = padded_seqs.clone()
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs_[i, :end, :] = seq
        padded_mask = torch.ones(len(sequences), max(lengths)).long()  ## padding index 1
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_mask[i, :end] = torch.zeros(end)


        return padded_seqs_, lengths, padded_mask

    def forward(self, nodes, context_rep, dia_triples, node_number): #nodes(247,100)
        path_list = []
        begin = 0
        dia_nodes = []
        for ind, num in enumerate(node_number):
            dia_node = nodes[begin:begin+num, :]  #取出一个对话的所有节点  (18,100)
            dia_node_trans = self.trans_node_to_decoder_dim(dia_node)  #(18,300)
            dia_nodes.append(dia_node_trans)
            begin += num

            #dia_node:当前对话对应的全部结点表示  dia_triples[ind]：当前对话对应的全部路径元组
            dia_path, dia_path_len = self.prepare_gru_input(dia_node, dia_triples[ind]) #dia_path:(路径总数,每条路径的最大结点个数,300) dia_path_len:[全1]
            dia_path_output, dia_path_hidden = self.path_encoder(dia_path, dia_path_len)  #使用GRU对路径编码

            #基于路径与对话上下文的注意力机制（不同路径的权重不一样）
            query = context_rep[ind].squeeze(0)
            dia_path_hidden_ = self.output_trans(dia_path_hidden)
            alpha = torch.mm(query, dia_path_hidden_.transpose(0, 1))  #（src_len,路径个数：10）
            alpha = F.softmax(alpha, dim=-1).sum(0) #tensor(10),计算每个路径的注意力权重
            dia_path_rep = torch.matmul(alpha.unsqueeze(0), dia_path_hidden_)
            dia_path_rep = self.layernorm(dia_path_rep)  #(1,300)
            path_list.append(dia_path_rep.squeeze(0))  #list16 [tensor(300),tensor(300),tensor(300)]

        path_feature = self.dropout(torch.stack(path_list))  #(16,300)
        node_feature, _, node_mask = self.merge_node(dia_nodes)  #node_feature(16,node2id_len,300)  node_mask(16,node2id_len)
        node_mask = node_mask.data.eq(config.PAD_idx).unsqueeze(1).cuda()

        return path_feature, node_feature, node_mask




