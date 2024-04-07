import os
import pickle
import torch
import torch.nn as nn
import json,math
from tqdm import tqdm
from utils.common_layer import TransKnowEncoder
from wordUtils import wordCate, Stack, lemmatize_all
from utils.config import args
from utils import config
import csv
from textrank4zh import TextRank4Keyword
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
from ast import literal_eval
from knowledge.utils.Embed import glove_embedding


PAD_idx = 0
UNK_idx = 1
EOS_idx = 2
SOS_idx = 3
SEP_idx = 4
SPK_idx = 5 #mask_speaker
LIS_idx = 6 #mask_listener
KG_idx = 7
CLS_idx = 8
PLS_idx = 9 #范例开始
PND_idx = 10 #范例结束

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




print("======loading EmpatheticDialogues======")
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


embeddings = glove_embedding(vocab)
encoder = TransKnowEncoder(args.emb_dim, args.hidden_dim, num_layers=1, num_heads=args.heads, total_key_depth=args.depth,
                       total_value_depth=args.depth, filter_size=args.filter, universal=args.universal)

# REMOVE_RELATIONS = ["Antonym", "ExternalURL", "NotDesires", "NotHasProperty", "NotCapableOf", "dbpedia", "DistinctFrom", "EtymologicallyDerivedFrom",
#                     "EtymologicallyRelatedTo", "SymbolOf", "FormOf", "AtLocation", "DerivedFrom", "CreatedBy", "Synonym", "MadeOf",
#                     "Reverse-Antonym", "Reverse-ExternalURL", "Reverse-NotDesires", "Reverse-NotHasProperty", "Reverse-NotCapableOf", "Reverse-dbpedia", "Reverse-DistinctFrom", "Reverse-EtymologicallyDerivedFrom",
#                     "Reverse-EtymologicallyRelatedTo", "Reverse-SymbolOf", "Reverse-FormOf", "Reverse-AtLocation", "Reverse-DerivedFrom",
#                     "Reverse-CreatedBy", "Reverse-Synonym", "Reverse-MadeOf"
#                     ]

REMOVE_RELATIONS = ["ExternalURL", "NotDesires", "NotHasProperty", "NotCapableOf", "dbpedia", "DistinctFrom",
                    "EtymologicallyDerivedFrom","EtymologicallyRelatedTo", "SymbolOf", "FormOf",
                    "AtLocation", "DerivedFrom", "SymbolOf", "CreatedBy", "Synonym", "MadeOf"]





def get_encode(token, sentence):
    word2index = vocab.word2index
    sentence = sentence.clone().detach().unsqueeze(0)
    mask = sentence.data.eq(config.PAD_idx).unsqueeze(1)

    token_emb = embeddings(torch.tensor([[word2index[token]]]))
    sentence_emb = embeddings(sentence)

    token_hidden = encoder(sentence_emb, token_emb, mask)

    return token_hidden[0][0]


def first_extract_triples(data):
    context = data['dialogue']
    context_id = {}
    for i, sentence in enumerate(context):
        for num, sgl in enumerate(sentence):
            if num == 0:  # 用来控制如果对话历史既有speaker又有listener时，将两个对话进行连接
                context_id[i] = [vocab.word2index[word] if word in vocab.word2index else config.UNK_idx for word in
                                 sgl]
            else:
                context_id[i] += [vocab.word2index[word] if word in vocab.word2index else config.UNK_idx for word in
                                  sgl]  # 得到句子的id表示
        if config.USE_CUDA:
            context_id[i] = torch.tensor(context_id[i])
    assert len(context) == len(context_id)
    # 将对话展平为一个列表形式
    context_dict = {}
    for i, text in enumerate(context):
        for j, sgl_text in enumerate(text):
            if j == 0:
                context_dict[i] = ' '.join(sgl_text)
            else:
                context_dict[i] = context_dict[i] + ' ' + ' '.join(sgl_text)
    # 找到每个对话中的关键概念（先通过textrank算法筛选出核心词汇，再进行动词、名词、形容词筛选）
    concept = json.load(open("/home/wangzikun/EmpEEK-GNN/knowledge/ConceptNet_ranked_dict.json", "r", encoding="utf-8"))
    word2index = vocab.word2index
    data['key_concepts'] = []
    data['triples'] = []
    for i, (sample, id) in tqdm(enumerate(zip(context_dict.values(), context_id.values())), total=len(context_dict)):
        concepts = {}  # concepts of each sample
        tr4w = TextRank4Keyword()
        tr4w.analyze(text=sample, lower=True, window=2)  # 文本小写，窗口为2
        if len(sample) >= 20:
            for item in tr4w.get_keywords(10, word_min_len=1):  # 每个对话中的关键词个数不超过10且每个的长度最小为1
                key = lemmatize_all(item.word)
                if key in concept and item.word in word2index:
                    concepts[key] = word2index[item.word]
                    # print(item.word, item.weight)
        elif len(sample) < 20 and len(sample) > 10:
            for item in tr4w.get_keywords(5, word_min_len=1):  # 每个对话中的关键词个数不超过10且每个的长度最小为1
                key = lemmatize_all(item.word)
                if key in concept and item.word in word2index:
                    concepts[key] = word2index[item.word]
        else:
            for item in tr4w.get_keywords(3, word_min_len=1):  # 每个对话中的关键词个数不超过10且每个的长度最小为1
                key = lemmatize_all(item.word)  # 词性还原是因为在concept当中的实体是词形还原过的
                if key in concept and item.word in word2index:  # 而这个词在上下文中是正常词表中的词
                    concepts[key] = word2index[item.word]
        concepts = dict(sorted(concepts.items(), key=lambda x: x[1], reverse=False))
        words_pos = nltk.pos_tag(concepts.keys())
        dialog_concepts = [
            word if word in word2index and word not in stop_words and wordCate(words_pos[wi]) else ''
            for wi, word in enumerate(concepts)]
        # 去除空的关键概念
        while '' in dialog_concepts:
            dialog_concepts.remove('')
        data['key_concepts'].append(dialog_concepts)

    return data





def first_create_context_concept(data):
    '''
    给定一个对话的emotion_s_txt
    :return:该txt的概念以及选取的三元组
    '''
    # print(os.getcwd())
    # with open('../prepare_data/all/valid_dataset_preproc.json', "r") as f:
    #     data_val = json.load(f)

    # data_tra = get_sample_data(data)

    #保存测试数据
    tst = first_extract_triples(data)
    with open('/home/wangzikun/EmpEEK-GNN/knowledge/data_key_concept/test.p', "wb") as ff2:
        pickle.dump(tst, ff2)
    print("test_data process successful")

    # 保存训练数据
    # tra = first_extract_triples(data)
    # with open('/home/wangzikun/EmpEEK-GNN/knowledge/data_key_concept/train.p', "wb") as ff:  #必须是”wb“模式，即二进制方式读取
    #     pickle.dump(tra, ff)
    # print("train_data process successful")

    # 保存验证数据
    # val = first_extract_triples(data)
    # with open('/home/wangzikun/EmpEEK-GNN/knowledge/data_key_concept/valid.p', "wb") as ff1:
    #     pickle.dump(val, ff1)
    # print("valid_data process successful")





#该py文件用于文本预处理，例如分割对话，提取关键词等
if __name__ == '__main__':
    print("Running main.....")
    # step 1
    # data_tra, data_val, data_tst, vocab = read_emp_dataset()  #只有4个字段的字典 # vocab = [word2index, word2count, index2word, n_words]
    # with open('EmpatheticDialogue/dataset_preproc.json', "w") as f:
    #     json.dump([data_tra, data_val, data_tst, vocab], f)
    #     print("Saved EmpatheticDialogue/dataset_preproc.json file.")

    # step 2
    # get_concept_dict()

    # step 3
    # rank_concept_dict()

    # step 4 save data
    # data_tra, data_val, data_tst, vocab = read_our_dataset()
    first_create_context_concept(data_tst)


    # with open('./cache_dialogue_2dim/kemp_train_new.p', "rb") as ff:
    #     data_train = pickle.load(ff)
    # with open('./cache_dialogue_2dim/kemp_valid_new.p', "rb") as ff1:
    #     data_valid = pickle.load(ff1)
    # with open('./cache_dialogue_2dim/kemp_test_new.p', "rb") as ff2:
    #     data_test = pickle.load(ff2)
    # print("===")