import os
import pickle
import torch
import json
from tqdm import tqdm
import csv
from textrank4zh import TextRank4Keyword
from src.common_layer import share_embedding
from ast import literal_eval
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()
from nltk.corpus import stopwords
stop_words = stopwords.words('english')


class GData:
    def __init__(self, init_index2word):
        self.word2index = {str(v): int(k) for k, v in init_index2word.items()}
        self.word2count = {str(v): 1 for k, v in init_index2word.items()}
        self.index2word = init_index2word
        self.n_words = len(init_index2word)  # Count default tokens
     #在这个类下定义的函数，可以用self直接访问
    def index_words(self, sentence):#通过传入的句子（由多个词组成），更新w2i，i2w，w2c，n_words
        for word in sentence:
            self.index_word(word.strip()) #22行

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def get_concept_dict():
    #读取训练集+验证集+测试集+词典
    with open('/home/wangzikun/EmpEEK-GNN/knowledge/emp_train_new.p', 'rb') as f1:
        data_tra = pickle.load(f1)

    with open('/home/wangzikun/EmpEEK-GNN/knowledge/emp_valid_new.p', 'rb') as f2:
        data_val = pickle.load(f2)

    with open('/home/wangzikun/EmpEEK-GNN/knowledge/emp_test_new.p', 'rb') as f3:
        data_tst = pickle.load(f3)

    with open('/home/wangzikun/EmpEEK-GNN/knowledge/vocab_preproc.p', 'rb') as f4:
        vocab = pickle.load(f4)

    word2index = vocab.word2index
    n_words = vocab.n_words








# if __name__ == '__main__':
    # first step:
    # get_concept_dict()












