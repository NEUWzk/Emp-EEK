import math,os
import numpy as np
from knowledge.utils import config
import torch
import torch.nn as nn

class Embeddings(nn.Module):
    def __init__(self,vocab, d_model, padding_idx=None):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model, padding_idx=padding_idx)
        # init.xavier_uniform(self.lut.weight)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


def gen_embeddings(vocab):
    """
        Generate an initial embedding matrix for `word_dict`.
        If an embedding file is not given or a word is not in the embedding file,
        a randomly initialized vector will be used.
    """
    embeddings = np.random.randn(vocab.n_words, config.emb_dim) * 0.01
    print('Embeddings: %d x %d' % (vocab.n_words, config.emb_dim))
    if config.emb_file is not None:
        print('Loading embedding file: %s' % config.emb_file)
        pre_trained = 0
        print(os.getcwd())
        for line in open(config.emb_file,encoding= 'utf-8').readlines():
            sp = line.split()
            if(len(sp) == config.emb_dim + 1):
                if sp[0] in vocab.word2index:
                    pre_trained += 1
                    embeddings[vocab.word2index[sp[0]]] = [float(x) for x in sp[1:]]
            else:
                print(sp[0])
        print('Pre-trained: %d (%.2f%%)' % (pre_trained, pre_trained * 100.0 / vocab.n_words))
    return embeddings

def glove_embedding(vocab, pretrain=True):
    #导入nn.Embedding
    embedding = Embeddings(vocab.n_words, config.emb_dim, padding_idx=config.PAD_idx)

    #更新嵌入矩阵为glove
    if(pretrain):
        #glove的嵌入矩阵（vocab.n_words,config.emb_dim）
        pre_embedding = gen_embeddings(vocab)
        embedding.lut.weight.data.copy_(torch.FloatTensor(pre_embedding))
        embedding.lut.weight.data.requires_grad = True
    return embedding