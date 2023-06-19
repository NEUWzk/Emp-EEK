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
LIS_idx = 6 #mask_listener
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

    def merge_concept(samples, samples_ext, samples_vads, samples_vad):
        #传进来四个参数：
        # item_info['concept'],每一项对应一个对话上下文：[ [],[],[con1,con2,con3]...] 和对话上下文长度一致
        # item_info['concept_ext'],
        # item_info["concept_vads"],
        # item_info["concept_vad"])
        concept_lengths = []  # 每个sample的concepts数目
        token_concept_lengths = []  # 每个sample的每个token的concepts数目
        concepts_list = []
        concepts_ext_list = []
        concepts_vads_list = []
        concepts_vad_list = []

        for i, sample in enumerate(samples):
            length = 0  # 记录当前样本总共有多少个concept，
            sample_concepts = []  #存放当前对话所有的concept 一维List
            sample_concepts_ext = []
            token_length = []
            vads = []
            vad = []

            for c, token in enumerate(sample):
                if token == []:  # 这个token没有concept
                    token_length.append(0)
                    continue
                length += len(token)
                token_length.append(len(token))
                sample_concepts += token
                sample_concepts_ext += samples_ext[i][c]
                vads += samples_vads[i][c]
                vad += samples_vad[i][c]

            if length > 10:
                # value, rank = torch.topk(torch.LongTensor(vad), k=10)
                value, rank = torch.topk(torch.Tensor(vad), k=10)  #按照每个概念的vad进行排序
                #rank:最后保留概念的下标索引
                new_length = 1
                new_sample_concepts = [SEP_idx]  # for each sample
                new_sample_concepts_ext = [SEP_idx]
                new_token_length = []
                new_vads = [[0.5, 0.0, 0.5]]
                new_vad = [0.0]

                cur_idx = 0
                for ti, token in enumerate(sample):
                    if token == []:
                        new_token_length.append(0)
                        continue
                    top_length = 0
                    for ci, con in enumerate(token):
                        point_idx = cur_idx + ci
                        if point_idx in rank:
                            top_length += 1
                            new_length += 1
                            new_sample_concepts.append(con)
                            new_sample_concepts_ext.append(samples_ext[i][ti][ci])
                            new_vads.append(samples_vads[i][ti][ci])
                            new_vad.append(samples_vad[i][ti][ci])
                            assert len(samples_vads[i][ti][ci]) == 3

                    new_token_length.append(top_length)
                    cur_idx += len(token)

                new_length += 1  # for sep token
                new_sample_concepts = [SEP_idx] + new_sample_concepts
                new_sample_concepts_ext = [SEP_idx] + new_sample_concepts_ext
                new_vads = [[0.5, 0.0, 0.5]] + new_vads
                new_vad = [0.0] + new_vad

                concept_lengths.append(new_length)  # the number of concepts including SEP
                token_concept_lengths.append(new_token_length)  # the number of tokens which have concepts
                concepts_list.append(new_sample_concepts)  #存放最终的概念idx
                concepts_ext_list.append(new_sample_concepts_ext)
                concepts_vads_list.append(new_vads)
                concepts_vad_list.append(new_vad)
                assert len(new_sample_concepts) == len(new_vads) == len(new_vad) == len(
                    new_sample_concepts_ext), "The number of concept tokens, vads [*,*,*], and vad * should be the same."
                assert len(new_token_length) == len(token_length)
            else:
                length += 1
                sample_concepts = [SEP_idx] + sample_concepts
                sample_concepts_ext = [SEP_idx] + sample_concepts_ext
                vads = [[0.5, 0.0, 0.5]] + vads
                vad = [0.0] + vad

                concept_lengths.append(length)
                token_concept_lengths.append(token_length)
                concepts_list.append(sample_concepts)
                concepts_ext_list.append(sample_concepts_ext)
                concepts_vads_list.append(vads)
                concepts_vad_list.append(vad)

        if max(concept_lengths) != 0:
            padded_concepts = torch.ones(len(samples),
                                         max(concept_lengths)).long()  ## padding index 1 (bsz, max_concept_len); add 1 for root
            padded_concepts_ext = torch.ones(len(samples),
                                             max(concept_lengths)).long()  ## padding index 1 (bsz, max_concept_len)
            padded_concepts_vads = torch.FloatTensor([[[0.5, 0.0, 0.5]]]).repeat(len(samples), max(concept_lengths),
                                                                                 1)  ## padding index 1 (bsz, max_concept_len)
            padded_concepts_vad = torch.FloatTensor([[0.0]]).repeat(len(samples),
                                                                    max(concept_lengths))  ## padding index 1 (bsz, max_concept_len)
            padded_mask = torch.ones(len(samples), max(concept_lengths)).long()  # concept(dialogue) state

            for j, concepts in enumerate(concepts_list):
                end = concept_lengths[j]
                if end == 0:
                    continue
                padded_concepts[j, :end] = torch.LongTensor(concepts[:end])
                padded_concepts_ext[j, :end] = torch.LongTensor(concepts_ext_list[j][:end])
                padded_concepts_vads[j, :end, :] = torch.FloatTensor(concepts_vads_list[j][:end])
                padded_concepts_vad[j, :end] = torch.FloatTensor(concepts_vad_list[j][:end])
                padded_mask[j, :end] = KG_idx  # for DIALOGUE STATE

            return padded_concepts, padded_concepts_ext, concept_lengths, padded_mask, token_concept_lengths, padded_concepts_vads, padded_concepts_vad
        else:  # there is no concept in this mini-batch
            return torch.Tensor([]), torch.LongTensor([]), torch.LongTensor([]), torch.BoolTensor([]), torch.LongTensor(
                []), torch.Tensor([]), torch.Tensor([])

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
        return adjacency



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
    assert context_batch.size(1) == context_vad_batch.size(1)

    concept_inputs = merge_concept(item_info['concept'],  #全是list
                                   item_info['concept_ext'],
                                   item_info["concept_vads"],#每一项是一个二维List,[ [],[], [[]] ]
                                   item_info["concept_vad"])

    concept_batch, concept_ext_batch, concept_lengths, mask_concept, token_concept_lengths,\
    concepts_vads_batch, concepts_vad_batch = concept_inputs

    if concept_batch.size()[0] != 0:
        adjacency_mask_batch = adj_mask(context_batch, context_lengths, concept_batch, token_concept_lengths)
    else:
        adjacency_mask_batch = torch.Tensor([])


    d = {}
    d["context_batch"] = context_batch.to(config.device)
    d["context_lengths"] = torch.LongTensor(context_lengths).to(config.device)
    d["mask_context"] = mask_context.to(config.device)
    d["context_vads"] = context_vads_batch.to(config.device)
    d["context_vad"] = context_vad_batch.to(config.device)

    #和concept相关的
    d["concept_batch"] = concept_batch.to(config.device)
    d["concept_ext_batch"] = concept_ext_batch.to(config.device)
    d["concept_lengths"] = torch.LongTensor(concept_lengths).to(config.device)
    d["mask_concept"] = mask_concept.to(config.device)
    d["concept_vads_batch"] = concepts_vads_batch.to(config.device)
    d["concept_vad_batch"] = concepts_vad_batch.to(config.device)
    d["adjacency_mask_batch"] = adjacency_mask_batch.bool().to(config.device)


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

    def __getitem__(self, index):
        item = {}
        item["dialogue"] = self.data["dialogue"][index]  # [ [],[] ]
        item["response"] = self.data["response"][index]  # []
        item["emotion"] = self.data["emotion"][index]  # str
        item["exemplars"] = self.data["exemplars"][index]  # [ [],[] ]
        item["empathy1_labels"] = self.data["empathy1_labels"][index]
        item["empathy2_labels"] = self.data["empathy2_labels"][index]  # int
        item["empathy3_labels"] = self.data["empathy3_labels"][index]
        item["emotion_widx"] = self.word2index[item["emotion"]]

        #转化为tensor后的数据
        item["dialogue2index"] = self.dialogue_to_index(item["dialogue"])
        item["response2index"] = self.response_to_index(item["response"])
        item["emotion2index"] = self.preprocess_emo(item["emotion"], self.emo_map)  # emo->int
        item["exemplars2index"] = self.exemplars_to_index(item["exemplars"])
        item["dialogue_mask"] = self.dialogue_to_mask(item["dialogue"])

        #接下来是和 VAD以及 concept相关的东西
        inputs = self.pro_concept([self.data["dialogue"][index],self.data["vads"][index],
                          self.data["vad"][index],self.data["concepts"][index]]) #这四个长度一致
        item["vads"], item["vad"], \
        item["concept_text"], item["concept"], item["concept_ext"], item["concept_vads"], \
        item["concept_vad"] = inputs

        assert len(item["dialogue2index"]) == len(item["dialogue_mask"]) == len(item["concept"])

        return item

    def preprocess_emo(self, emotion, emo_map): #emotion:str
        if emotion == "":
            return emo_map['null']
        else:
            return emo_map[emotion] #int索引


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
        return all_exemplars_tokens#必须保证每一个列表的长度一致，否则无法转tensor

    def dialogue_to_mask(self,context): #区分说话者、倾听者
        X_mask = [CLS_idx]
        for i, sentence in enumerate(context):
            spk = self.vocab.word2index["SPK"] if i % 2 == 0 else self.vocab.word2index["LIS"]
            X_mask += [spk for _ in range(len(sentence))]  # 说话者信息的mask，有多少个token就有多少个数字
            # if len(context) != 1:
            #     X_mask += [SEP_idx]
        # X_mask += [EOS_idx]
        return torch.LongTensor(X_mask)

    def pro_concept(self,arr):
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
    train_file = "../knowledge/kemp_train_new.p"
    valid_file = "../knowledge/kemp_valid_new.p"
    test_file = "../knowledge/kemp_test_new.p"
    vocab_file = "../knowledge/vocab_preproc.p"
    with open(train_file, 'rb') as f1:
        data_tra = pickle.load(f1)
    with open(valid_file, 'rb') as f2:
        data_val = pickle.load(f2)
    with open(test_file, 'rb') as f3:
        data_tst = pickle.load(f3)
    with open(vocab_file, 'rb') as f4:
        vocab = pickle.load(f4)

    return data_tra,data_val,data_tst,vocab

def prepare_data_seq1(batch_size=32):
    pairs_tra, pairs_val, pairs_tst, vocab = load_dataset()
    dataset_train = MyDataset(pairs_tra,vocab)
    dataset_valid = MyDataset(pairs_val, vocab)
    dataset_test = MyDataset(pairs_tst, vocab)

    data_loader_tra = torch.utils.data.DataLoader(dataset=dataset_train,
                                                  batch_size=16,
                                                  shuffle=True, collate_fn=collate_fn)
    data_loader_val = torch.utils.data.DataLoader(dataset=dataset_valid,
                                                  batch_size=16,
                                                  shuffle=True, collate_fn=collate_fn)
    data_loader_tst = torch.utils.data.DataLoader(dataset=dataset_test,
                                                  batch_size=1,
                                                  shuffle=False, collate_fn=collate_fn)

    return data_loader_tra, data_loader_val, data_loader_tst, vocab, len(dataset_train.emo_map)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)











