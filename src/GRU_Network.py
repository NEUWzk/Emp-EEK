import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers = 2, bidirectional = False):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_directions = 2 if bidirectional else 1
        self.gru = nn.GRU(input_size, hidden_size, n_layers, bidirectional= bidirectional, batch_first= True)
        self.linear = nn.Linear(hidden_size*2, hidden_size)
    def __init__hidden(self,batch_size):
        hidden = torch.zeros(self.n_layers*self.n_directions, batch_size, self.hidden_size)
        return hidden.cuda()
    def forward(self, input, seq_lengths):  #input:(每个对话有10条路径,经过的最大结点个数,)
        batch_size = input.size(0)  #10个边（keyword-knowledge之间的关系数）
        hidden = self.__init__hidden(batch_size)
        gru_input = pack_padded_sequence(input, seq_lengths, batch_first= True, enforce_sorted=False)
        # batch_output = []
        output, hidden = self.gru(gru_input, hidden)
        # for ind, len in enumerate(seq_lengths):
        #     batch_output.append(output[ind, len-1, : ])
        # output = torch.stack(batch_output)
        # output = self.linear(output)
        if self.n_directions == 2:  # 双向的，则需要拼接起来
            hidden_cat = torch.cat([hidden[-1], hidden[-2]], dim=1)  #tensor(10，600) -> (路径条数，600)
        else:
            hidden_cat = hidden[-1]  # 单向的，则不用处理
        return output, hidden_cat