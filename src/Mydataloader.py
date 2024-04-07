import os
import pickle
import torch
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
import config


PAD_idx = 1
UNK_idx = 0
EOS_idx = 2
SOS_idx = 3
SEP_idx = 4
SPK_idx = 5 #mask_speaker
LIS_idx = 6 #mask_listene
KG_idx = 7
CLS_idx = 8
PLS_idx = 9 #范例开始
PND_idx = 10 #范例结束



def collate_fn(batch_data):
    def merge(sequences): #用于pad序列
        lengths = [len(seq) for seq in sequences] #得到每个tensor的长度
        padded_seqs = torch.ones(len(sequences), max(lengths)).long()  ## padding index 1 [32,69]
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end] #用实际的tensor去覆盖mask矩阵
        return padded_seqs, lengths

    def merge_exemplars(sequences): #[ [tensor,tensor],[],[],[] .. ]
        lengths = []
        exemplar_representations = []
        for i in range(len(sequences)): #batch大小
            li = []
            for item in sequences[i]: #tensor
                tensor_len = len(item)
                li.append(tensor_len)
            lengths.append(li)   #[[],[],[]] 记录所有范例句子的长度
        max_list = [] #长度为 batch
        for item in lengths: #[]
            max_len = max(item)
            max_list.append(max_len)

        for idx, seq in enumerate(sequences): #sequences:[ [tensor,tensor],[],[],[] .. ]
            padded_seqs = torch.ones(len(seq), max_list[idx]).long() #(10,max_len)
            for tensor_idx,exemplar_tensor in enumerate(seq):
                end = padded_seqs[tensor_idx] #[第i行pad矩阵]
                end[:len(exemplar_tensor)] = exemplar_tensor
            exemplar_representations.append(padded_seqs)
        return exemplar_representations, lengths


    def merge_vad(vads_sequences, vad_sequences):  # for context
        lengths = [len(seq) for seq in vad_sequences]
        padding_vads = torch.FloatTensor([[[0.5, 0.0, 0.5]]]).repeat(len(vads_sequences), max(lengths), 1)
        padding_vad = torch.FloatTensor([[0.5]]).repeat(len(vads_sequences), max(lengths))

        for i, vads in enumerate(vads_sequences):
            end = lengths[i]  # the length of context
            padding_vads[i, :end, :] = torch.FloatTensor(vads[:end])
            padding_vad[i, :end] = torch.FloatTensor(vad_sequences[i][:end])
        return padding_vads, padding_vad

    def adj_mask(context, context_lengths, concepts, token_concept_lengths):
        '''

        :param self:
        :param context: (bsz, max_context_len)
        :param context_lengths: [] len=bsz
        :param concepts: (bsz, max_concept_len)
        :param token_concept_lengths: [] len=bsz;
        :return:
        '''
        bsz, max_context_len = context.size()
        max_concept_len = concepts.size(1)  # include sep token
        adjacency_size = max_context_len + max_concept_len
        adjacency = torch.ones(bsz, max_context_len, adjacency_size)  ## todo padding index 1, 1=True

        for i in range(bsz):
            # ROOT -> TOKEN
            adjacency[i, 0, :context_lengths[i]] = 0
            adjacency[i, :context_lengths[i], 0] = 0

            con_idx = max_context_len + 1  # add 1 because of sep token
            for j in range(context_lengths[i]):
                adjacency[i, j, j - 1] = 0  # TOEKN_j -> TOKEN_j-1

                token_concepts_length = token_concept_lengths[i][j]
                if token_concepts_length == 0:
                    continue
                else:
                    adjacency[i, j, con_idx:con_idx + token_concepts_length] = 0
                    adjacency[i, 0, con_idx:con_idx + token_concepts_length] = 0
                    con_idx += token_concepts_length



    #把dialogue按照长度降序排列
    batch_data.sort(key=lambda x: len(x["dialogue2index"]), reverse=True)  #[data1,data2...]
    item_info = {}   #'context':16条数据  'response':16条数据
    for key in batch_data[0].keys():  #将一个batch_size大小的数据放到同一个key下面
        item_info[key] = [d[key] for d in batch_data]

    assert len(item_info['dialogue']) == len(item_info['vad'])

    context_batch, context_lengths = merge(item_info['dialogue2index'])  #padding,此处没有排序
    target_batch, target_lengths = merge(item_info['response2index'])
    mask_context, input_mask_lengths = merge(item_info['dialogue_mask'])
    exem_batch, exem_lengths = merge_exemplars(item_info['exemplars2index'])

    context_vads_batch, context_vad_batch = merge_vad(item_info['vads'], item_info['vad'])
    # assert context_batch.size(1) == context_vad_batch.size(1)


    #开始构建GNN相关内容
    node_batch, node_lengths = merge(item_info['node_id'])
    node_type_batch, node_type_length = merge(item_info['node_type'])



    d = {}
    d["context_batch"] = context_batch.to(config.device)
    d["context_lengths"] = torch.LongTensor(context_lengths).to(config.device)
    d["mask_context"] = mask_context.to(config.device)
    d["context_vads"] = context_vads_batch.to(config.device)
    d["context_vad"] = context_vad_batch.to(config.device)

    #和concept相关的
    # d["concepts_batch"] = [torch.LongTensor(item).to(config.device) for item in item_info['key_concepts']]  #关键词
    d["node_number"] = torch.LongTensor(node_lengths).to(config.device)
    d["node_type"] = node_type_batch.to(config.device)
    d["node_id"] = node_batch.to(config.device)
    d['final_dict'] = item_info["final_dict"]
    d["node2id"] = item_info['node2id']
    d['keyword2id'] = item_info['keyword2id']
    d["concepts_batch"] = item_info['key_concepts']
    d["dia_triples"] = item_info['dia_triples']

    # output相关的
    d["target_batch"] = target_batch.to(config.device)
    d["target_lengths"] = torch.LongTensor(target_lengths).to(config.device)

    # exem + emotion
    d["emotion_widx"] = torch.LongTensor(item_info['emotion_widx']).to(config.device)
    d["emotion_label"] = torch.LongTensor(item_info['emotion2index']).to(config.device)
    d["empathy1_labels"] = torch.LongTensor(item_info['empathy1_labels']).to(config.device)
    d["empathy2_labels"] = torch.LongTensor(item_info['empathy2_labels']).to(config.device)
    d["empathy3_labels"] = torch.LongTensor(item_info['empathy3_labels']).to(config.device)
    d["exem_batch"] = exem_batch
    for i in range(len(d["exem_batch"])):
        d["exem_batch"][i] = d["exem_batch"][i].to(config.device)

    d["context_txt"] = item_info["dialogue"]
    d["target_txt"] = item_info['response']
    d["emotion_txt"] = item_info['emotion']
    d["exemplars_txt"] = item_info['exemplars']
    d["concept_txt"] = item_info['concept_text']

    # empathy_labels processing
    empathy1_label = item_info['empathy1_labels']  #list
    empathy2_label = item_info['empathy2_labels']
    empathy3_label = item_info['empathy3_labels']

    assert len(empathy1_label) == len(empathy2_label) == len(empathy3_label)
    empathy_len = len(empathy2_label)
    empathy_list = []
    for i in range(empathy_len):
        empathy_list.append([empathy1_label[i],empathy2_label[i],empathy3_label[i]])
    d["empathy_label_123"] = torch.LongTensor(empathy_list).to(config.device) #(batch,3)

    emo_label = d["emotion_label"].cpu().numpy().tolist()  # list
    list1 = []
    for item1 in emo_label:  # int
        li = [0] * 32  # [0 0 0 0...]  one-hot
        li[item1] = 1
        list1.append(li)

    d["target_program"] = list1
    d["empathy_label"] = d["empathy_label_123"].cpu().numpy().tolist()

    return d



class MyDataset(Dataset):
    def __init__(self, data, vocab):
        self.vocab = vocab
        self.data = data
        self.word2index = vocab.word2index
        self.emo_map = {
            'surprised': 0, 'excited': 1, 'annoyed': 2, 'proud': 3, 'angry': 4, 'sad': 5, 'grateful': 6,
            'lonely': 7, 'impressed': 8, 'afraid': 9, 'disgusted': 10, 'confident': 11, 'terrified': 12,
            'hopeful': 13, 'anxious': 14, 'disappointed': 15, 'joyful': 16, 'prepared': 17, 'guilty': 18, 'furious': 19,
            'nostalgic': 20, 'jealous': 21, 'anticipating': 22, 'embarrassed': 23, 'content': 24, 'devastated': 25,
            'sentimental': 26, 'caring': 27, 'trusting': 28, 'ashamed': 29, 'apprehensive': 30, 'faithful': 31

        }
        self.map_emo = {0: 'surprised', 1: 'excited', 2: 'annoyed', 3: 'proud',
                        4: 'angry', 5: 'sad', 6: 'grateful', 7: 'lonely', 8: 'impressed',
                        9: 'afraid', 10: 'disgusted', 11: 'confident', 12: 'terrified',
                        13: 'hopeful', 14: 'anxious', 15: 'disappointed', 16: 'joyful',
                        17: 'prepared', 18: 'guilty', 19: 'furious', 20: 'nostalgic',
                        21: 'jealous', 22: 'anticipating', 23: 'embarrassed', 24: 'content',
                        25: 'devastated', 26: 'sentimental', 27: 'caring', 28: 'trusting',
                        29: 'ashamed', 30: 'apprehensive', 31: 'faithful'}


    def __len__(self):
        return len(self.data['emotion'])

    def merge_concept(self, external_concept, samples_vads, samples_vad, key_concepts):
        assert len(external_concept) == len(samples_vads) == len(samples_vad) == len(key_concepts)
        concept_lengths = len(key_concepts)  #当前的关键词个数
        #external_concept表示初始的外部概念个数，一般大于10个， key_concepts是当前对话提取到的关键词
        token_length = []
        sample_concepts = [] #将所有候选的外部知识展开，放在该列表
        vad, vads = [], []
        final_dict = {}
        length = 0  # 记录当前样本总共有多少个concept
        for c, token in enumerate(external_concept):
            if token == []:
                token_length.append(0)
                continue
            length += len(token)  #当前关键词对应的外部概念列表非空
            token_length.append(len(token)) #记录每个关键词有几个外部知识
            sample_concepts += token
            vads += samples_vads[c]
            vad += samples_vad[c]

        if length > 10:  #当前sample_concepts有20个候选的外部概念，要进行筛选
            value, rank = torch.topk(torch.Tensor(vad), k=10)  # 按照每个概念的vad进行排序

            new_length = 0
            new_sample_concepts = []  #筛选后的外部知识集合
            new_token_length = []
            new_vads = []
            new_vad = []

            cur_idx = 0
            for ti, token in enumerate(external_concept):
                if token == []:
                    new_token_length.append(0)
                    continue

                top_length = 0
                cur_concept = []  # 用于追加输入最终概念的列表
                for ci, con in enumerate(token):
                    point_idx = cur_idx + ci  # 当前概念位于原始的所有的外部知识中的位置
                    if point_idx in rank:
                        cur_concept.append(con)  #添加到最终的外部知识集合
                        final_dict[key_concepts[ti]] = cur_concept
                        top_length += 1
                        new_length += 1
                        new_sample_concepts.append(con)
                        new_vads.append(samples_vads[ti][ci])
                        new_vad.append(samples_vad[ti][ci])
                        assert len(samples_vads[ti][ci]) == 3

                new_token_length.append(top_length)  #当前单词所对的3个概念保留了几个
                cur_idx += len(token)

            sample_concepts = new_sample_concepts


        else:  #如果候选的外部知识不足10
            length += 1
            sample_concepts = [SEP_idx]
            vads = [[0.5, 0.0, 0.5]]
            vad = [0.0]

            token_length.append(length)
            vads.append(vads)
            vad.append(vad)

        # new_sample_concepts是最终保留的10个外部知识的index
        return final_dict, sample_concepts

    def __getitem__(self, index):
        item = {}
        item["context_g_text"] = self.data["key_concepts"][index]  #关键词 [a,b,c,d...]
        item["dialogue"] = self.data["dialogue"][index]  # [ [],[] ]
        item["response"] = self.data["response"][index]  # []
        item["emotion"] = self.data["emotion"][index]  # str
        item["exemplars"] = self.data["exemplars"][index]  # [ [],[] ]
        item["empathy1_labels"] = self.data["empathy1_labels"][index]
        item["empathy2_labels"] = self.data["empathy2_labels"][index]  # int
        item["empathy3_labels"] = self.data["empathy3_labels"][index]
        item["emotion_widx"] = self.word2index[item["emotion"]]
        item["key_concepts_knowledge"] = self.data["key_concepts_knowledge"][index]

        #转化为tensor后的数据
        item["dialogue2index"] = self.dialogue_to_index(item["dialogue"])
        item["response2index"] = self.response_to_index(item["response"])
        item["emotion2index"] = self.preprocess_emo(item["emotion"], self.emo_map)  # emo->int
        item["exemplars2index"] = self.exemplars_to_index(item["exemplars"])
        item["dialogue_mask"] = self.dialogue_to_mask(item["dialogue"])


        #接下来是和 VAD以及 concept相关的东西,假设当前句子有9个关键词，那么返回的长度应该为10（添加了头部标记）
        inputs = self.pro_concept([[self.data["key_concepts"][index]],self.data["vads"][index],
                          self.data["vad"][index],self.data["key_concepts_knowledge"][index]])
        item["vads"], item["vad"], \
        item["concept_text"], item["concept"], item["concept_ext"], item["concept_vads"], \
        item["concept_vad"] = inputs  #vads:[V,A,D]   vad:经过计算后的，只有一维的数字

        assert len(item["vads"]) == len(item["vad"]) == len(item["concept"])  #(关键词个数+1)

        item["key_concepts"] = self.concept_to_index(item["context_g_text"])  #关键词 -> index
        final_dict, final_external_knowledge = self.merge_concept(item['concept'][1:],  #[]
                      item["concept_vads"][1:],  #[]
                      item["concept_vad"][1:],   #[]
                      item["key_concepts"])  #[]


        item["node_number"], item['node_type'],item['node2id'], item['keyword2id'], item["node_id"] = \
            self.preprocess_node(final_dict, item["context_g_text"], item["key_concepts"])
        item["final_dict"] = self.process_dict(final_dict)
        item['dia_triples'] = self.process_triples(item["final_dict"], item['node2id'], item['keyword2id'])
        return item

    def process_triples(self,final_dict, node2id, keyword2id):  #final_dict有时候是空的
        tupless = []
        if len(final_dict) != 0:
            for each_key in final_dict.keys():  # work
                each_key_index = node2id[each_key]  # 0
                values_list = final_dict[each_key]  # [play,pain]
                for each_value in values_list:
                    value_index = node2id[each_value]  # 1
                    tupless.append((each_key_index, value_index))
        else: #字典为空
            if len(keyword2id) != 0 :
                for p in keyword2id:  #比如索引99的，只有一个关键词，final_dict = {}
                    for q in keyword2id:  #p,q均为['trusting']
                        tupless.append((keyword2id[p],keyword2id[q]))  #防止tuple为空，后续计算出现错误
        return tupless


    def process_dict(self,final_dict):
        new_final_dict = {}
        for i in final_dict.keys(): #254
            cur_word = self.vocab.index2word[i]
            concept_lis = [self.vocab.index2word[each_index] for each_index in final_dict[i]]
            new_final_dict[cur_word] = concept_lis
        return new_final_dict



    def preprocess_node(self, node_dict, key_concepts_words, key_concepts_index):
        node_id = []
        node_type = []
        new_node_id = []
        node2id = {}
        keyword2id = {}
        dia_triples = []  #tuple(a,b)
        # 统计所有出现的node_id，借助set去重后存入new_node_id列表
        for index in key_concepts_index: #先把所有的关键词放到node_id里面
            node_id += [index]
            if index in node_dict.keys():
                node_id += node_dict[index]
                new_node_id = list(set(node_id))  #去重复
                new_node_id.sort(key=node_id.index)
            else:
                new_node_id = list(set(node_id))  # 去重复
                new_node_id.sort(key=node_id.index)


        new_node_id_len = 0
        for each_node in new_node_id:  #each_node:84
            cur_word = self.vocab.index2word[each_node]
            if cur_word not in node2id:
                node2id[cur_word] = new_node_id_len
                new_node_id_len += 1
            if each_node in key_concepts_index:
                keyword2id[cur_word] = node2id[cur_word]  #外部概念可能跟关键词重叠了，因此必须用len(node2id)作为结点长度
                node_type.append(0)
            else:
                node_type.append(1)

        return len(node2id), torch.LongTensor(node_type), node2id, keyword2id, torch.LongTensor(new_node_id)



    def preprocess_emo(self, emotion, emo_map): #emotion:str
        if emotion == "":
            return emo_map['null']
        else:
            return emo_map[emotion] #int索引


    def concept_to_index(self,concept):
        X_dial = []
        for i, word in enumerate(concept):
            X_dial += [self.vocab.word2index[word] if word in self.vocab.word2index else config.UNK_idx]
        return X_dial


    def dialogue_to_index(self,dialogue):  #[ [],[] ]
        X_dial = [CLS_idx] #[cls]
        for idx, sentence in enumerate(dialogue): #sentence:[]
            X_dial += [self.vocab.word2index[word] if word in self.vocab.word2index
                       else UNK_idx for word in sentence]
            # if len(dialogue) != 1:
            #     X_dial += [SEP_idx]  #多轮对话之间加[SEP]标志
        # X_dial += [EOS_idx] #[int,int...] ，每组多轮对话变成了一个列表
        return torch.LongTensor(X_dial)

    def response_to_index(self,response): #[]
        res = [self.vocab.word2index[word] if word in self.vocab.word2index
                    else UNK_idx for word in response] + [EOS_idx]
        return torch.LongTensor(res)


    def exemplars_to_index(self,exemplars): #[ [],[] ]
        all_exemplars_tokens = []
        for idx, sentence in enumerate(exemplars): #[]
            X_dial = [PLS_idx]
            X_dial += [self.vocab.word2index[word] if word in self.vocab.word2index
                       else UNK_idx for word in sentence]
            X_dial += [PND_idx]
            X_dial_tensor = torch.LongTensor(X_dial)#[ [],[] ] 每行都是这样,
            all_exemplars_tokens.append(X_dial_tensor)
        return all_exemplars_tokens   #必须保证每一个列表的长度一致，否则无法转tensor

    def dialogue_to_mask(self,context): #区分说话者、倾听者
        X_mask = [CLS_idx]
        for i, sentence in enumerate(context):
            spk = self.vocab.word2index["SPK"] if i % 2 == 0 else self.vocab.word2index["LIS"]
            X_mask += [spk for _ in range(len(sentence))]  # 说话者信息的mask，有多少个token就有多少个数字
        return torch.LongTensor(X_mask)

    def pro_concept(self, arr):
        context = arr[0]
        context_vads = arr[1]
        context_vad = arr[2]
        concept = [arr[3][l][0] for l in range(len(arr[3]))]
        concept_vads = [arr[3][l][1] for l in range(len(arr[3]))]
        concept_vad = [arr[3][l][2] for l in range(len(arr[3]))]

        X_vads = [[0.5, 0.0, 0.5]]
        X_vad = [0.0]
        X_concept_text = defaultdict(list)
        X_concept = [[]]  # 初始值是cls token
        X_concept_ext = [[]]
        X_concept_vads = [[0.5, 0.0, 0.5]]
        X_concept_vad = [0.0]
        assert len(context) == len(concept)

        for i, sentence in enumerate(context):
            X_vads += context_vads[i]
            X_vad += context_vad[i]
            for j, token_conlist in enumerate(concept[i]):
                if token_conlist == []:
                    X_concept.append([])
                    X_concept_ext.append([])
                    X_concept_vads.append([0.5, 0.0, 0.5])
                    X_concept_vad.append(0.0)
                else:
                    X_concept_text[sentence[j]] += token_conlist[:3]  #concept个数
                    X_concept.append(
                        [self.word2index[con_word] if con_word in self.word2index else UNK_idx
                         for con_word in token_conlist[:3]])

                    con_ext = []
                    for con_word in token_conlist[:3]:
                        if con_word in self.word2index:
                            con_ext.append(self.word2index[con_word])
                        else:
                            con_ext.append(UNK_idx)
                    X_concept_ext.append(con_ext)
                    X_concept_vads.append(concept_vads[i][j][:3])
                    X_concept_vad.append(concept_vad[i][j][:3])

                    assert len(
                        [self.word2index[con_word] if con_word in self.word2index else UNK_idx for con_word in
                         token_conlist[:3]]) == len(concept_vads[i][j][:3]) == len(concept_vad[i][j][:3])


        return  X_vads, X_vad, X_concept_text, X_concept, X_concept_ext, X_concept_vads, X_concept_vad






def flatten(t):
    return [item for sublist in t for item in sublist]

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


def load_dataset():
    train_file = "/root/Emp-EGP/knowledge/data_key_concept/train_key_concept_new1.p"  #记得替换为key_concepts完整版的
    valid_file = "/root/Emp-EGP/knowledge/data_key_concept/valid_key_concept_new1.p"
    test_file = "/root/Emp-EGP/knowledge/data_key_concept/test_key_concept_new1.p"

    # train_file = "/home/wangzikun/concept-result/train_new.p"
    # valid_file = "/home/wangzikun/concept-result/valid_new.p"
    # test_file = "/home/wangzikun/concept-result/test_new.p"
    vocab_file = "/root/Emp-EGP/knowledge/vocab_preproc.p"
    with open(train_file, 'rb') as f1:
        data_tra = pickle.load(f1)
    with open(valid_file, 'rb') as f2:
        data_val = pickle.load(f2)
    with open(test_file, 'rb') as f3:
        data_tst = pickle.load(f3)
    with open(vocab_file, 'rb') as f4:
        vocab = pickle.load(f4)

    return data_tra,data_val,data_tst,vocab

def prepare_data_seq1(batch_size):
    pairs_tra, pairs_val, pairs_tst, vocab = load_dataset()
    dataset_train = MyDataset(pairs_tra,vocab)
    dataset_valid = MyDataset(pairs_val, vocab)
    dataset_test = MyDataset(pairs_tst, vocab)

    data_loader_tra = torch.utils.data.DataLoader(dataset=dataset_train,
                                                  batch_size=batch_size,
                                                  shuffle=True, collate_fn=collate_fn)
    data_loader_val = torch.utils.data.DataLoader(dataset=dataset_valid,
                                                  batch_size=batch_size,
                                                  shuffle=True, collate_fn=collate_fn)
    data_loader_tst = torch.utils.data.DataLoader(dataset=dataset_test,
                                                  batch_size=1,
                                                  shuffle=False, collate_fn=collate_fn)

    return data_loader_tra, data_loader_val, data_loader_tst, vocab, len(dataset_train.emo_map)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)










