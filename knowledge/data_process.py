import numpy as np
import torch
import json,pickle
import csv
import nltk
nltk.download('punkt')
from ast import literal_eval
from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
from nltk.corpus import wordnet


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



def gen_embeddings(n_words, word2index, emb_dim=300, emb_file='glove.6B.300d.txt'):
    """
        Generate an initial embedding matrix for `word_dict`.
        If an embedding file is not given or a word is not in the embedding file,
        a randomly initialized vector will be used.
    """
    embeddings = np.random.randn(n_words, emb_dim) * 0.01
    print('Embeddings: %d x %d' % (n_words, emb_dim))
    if emb_file is not None:
        print('Loading embedding file: %s' % emb_file)
        pre_trained = 0
        for line in open(emb_file).readlines():
            sp = line.split()
            if(len(sp) == emb_dim + 1):
                if sp[0] in word2index:
                    pre_trained += 1
                    embeddings[word2index[sp[0]]] = [float(x) for x in sp[1:]]
            else:
                print(sp[0])
        print('Pre-trained: %d (%.2f%%)' % (pre_trained, pre_trained * 100.0 / n_words))
    return embeddings


def emotion_intensity(NRC, word):
    '''
    Function to calculate emotion intensity (Eq. 1 in our paper)
    :param NRC: NRC_VAD vectors
    :param word: query word
    :return:
    '''
    v, a, d = NRC[word]
    a = a/2
    return (np.linalg.norm(np.array([v, a]) - np.array([0.5, 0])) - 0.06467)/0.607468


def get_concept_dict():
    '''
    Retrieve concepts from ConceptNet using the EmpatheticDialogue tokens as queries
    :return:
    '''

    with open('./train_tokens.p', 'rb') as f1:
        data_tra = pickle.load(f1)

    with open('./valid_tokens.p', 'rb') as f2:
        data_val = pickle.load(f2)

    with open('./test_tokens.p', 'rb') as f3:
        data_tst = pickle.load(f3)

    with open('./vocab_preproc.p', 'rb') as f4:
        vocab = pickle.load(f4)

    word2index = vocab.word2index
    n_words= vocab.n_words

    embeddings = gen_embeddings(n_words, word2index) #(vocab_len,300) 词嵌入

    VAD = json.load(open("VAD.json", "r", encoding="utf-8"))  # NRC_VAD
    CN = csv.reader(open("assertions.csv", "r", encoding="utf-8"))  # ConceptNet raw file

    concept_dict = {}
    concept_file = open("ConceptNet1.json", "w", encoding="utf-8")

    relation_dict = {}
    rd = open("relation1.json", "w", encoding="utf-8")

    for i, row in enumerate(CN):
        if i%1000000 == 0:
            print("Processed {} rows".format(i)) #按照tab对原始的行进行分割，共有5个元素
        items = "".join(row).split("\t")
        c1_lang = items[2].split("/")[2]
        c2_lang = items[2].split("/")[2]
        if c1_lang == "en" and c2_lang == "en": #语言类型
            if len(items) != 5:
                print("concept error!")
            relation = items[1].split("/")[2] #Antonmy:反义词
            c1 = items[2].split("/")[3]   #左实体
            c2 = items[3].split("/")[3]   #右实体
            c1 = wnl.lemmatize(c1)
            c2 = wnl.lemmatize(c2)
            weight = literal_eval("{" + row[-1].strip())["weight"]

            if weight < 1.0:  # filter tuples where confidence score is smaller than 1.0
                continue
            if c1 in word2index and c2 in word2index and c1 != c2 and c1.isalpha() and c2.isalpha():
                if relation not in word2index:
                    if relation in relation_dict: #添加到关系字典里
                        relation_dict[relation] += 1 #统计次数？
                    else:
                        relation_dict[relation] = 0  #{'Antonym':0}
                c1_vector = torch.Tensor(embeddings[word2index[c1]]) #C1 C2都是str，计算embedding
                c2_vector = torch.Tensor(embeddings[word2index[c2]])
                c1_c2_sim = torch.cosine_similarity(c1_vector, c2_vector, dim=0).item()

                v1, a1, d1 = VAD[c1] if c1 in VAD else [0.5, 0.0, 0.5] #获取C1的VAD向量
                v2, a2, d2 = VAD[c2] if c2 in VAD else [0.5, 0.0, 0.5]
                emotion_gap = 1-(abs(v1-v2) + abs(a1-a2))/2  #C1和C2之间的情感距离（使用VAD向量计算的）
                # <c1 relation c2>
                if c2 not in stop_words:
                    c2_vad = emotion_intensity(VAD, c2) if c2 in VAD else 0.0  #得分值
                    # score = c2_vad + c1_c2_sim + (weight - 1) / (10.0 - 1.0) + emotion_gap
                    score = c2_vad + emotion_gap  #得分
                    if c1 in concept_dict: #初始为{}
                        concept_dict[c1][c2] = [relation, c2_vad, c1_c2_sim, weight, emotion_gap, score]
                    else:
                        concept_dict[c1] = {} #{'abandon':{'acquire':[]}}
                        concept_dict[c1][c2] = [relation, c2_vad, c1_c2_sim, weight, emotion_gap, score]
                # <c2 relation c1>
                if c1 not in stop_words:
                    c1_vad = emotion_intensity(VAD, c1) if c1 in VAD else 0.0
                    # score = c1_vad + c1_c2_sim + (weight - 1) / (10.0 - 1.0) + emotion_gap
                    score = c1_vad + emotion_gap
                    if c2 in concept_dict:   #这里比GP的要多几个,比如emotion_gap
                        concept_dict[c2][c1] = [relation, c1_vad, c1_c2_sim, weight, emotion_gap, score]
                    else:
                        concept_dict[c2] = {}
                        concept_dict[c2][c1] = [relation, c1_vad, c1_c2_sim, weight, emotion_gap, score]
                    #concept_dict是一个嵌套字典，字典的每一项 ：“abandon”:{'acquire':[]}
    print("concept num: ", len(concept_dict)) #15283
    json.dump(concept_dict, concept_file)  #将处理好的字典保存到conceptNet.json文件中

    relation_dict = sorted(relation_dict.items(), key=lambda x: x[1], reverse=True)
    json.dump(relation_dict, rd)  #将处理好的关系保存到relation.json文件中
    print("ConceptNet处理完毕...")




def rank_concept_dict(): #对于每一个word,把和他相关的概念进行排序
    concept_dict = json.load(open("ConceptNet1.json", "r", encoding="utf-8"))
    rank_concept_file = open('ConceptNet_VAD_dict1.json', 'w', encoding='utf-8')  #排名后的ConceptNet字典

    rank_concept = {}
    for i in concept_dict:
        # [relation, c1_vad, c1_c2_sim, weight, emotion_gap, score]   relation, weight, score

        rank_concept[i] = dict(sorted(concept_dict[i].items(), key=lambda x: x[1][5], reverse=True))
        rank_concept[i] = [[l, concept_dict[i][l][0], concept_dict[i][l][1], concept_dict[i][l][2], concept_dict[i][l][3], concept_dict[i][l][4], concept_dict[i][l][5]] for l in concept_dict[i]]
    json.dump(rank_concept, rank_concept_file, indent=4)
    print("succeed exec ConceptNet_VAD_dict1.json!")  #排名后的Json文件


REMOVE_RELATIONS = ["Antonym", "ExternalURL", "NotDesires", "NotHasProperty", "NotCapableOf", "dbpedia", "DistinctFrom", "EtymologicallyDerivedFrom",
                    "EtymologicallyRelatedTo", "SymbolOf", "FormOf", "AtLocation", "DerivedFrom", "SymbolOf", "CreatedBy", "Synonym", "MadeOf"]

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def wordCate(word_pos):
    w_p = get_wordnet_pos(word_pos[1])
    if w_p == wordnet.NOUN or w_p == wordnet.ADV or w_p == wordnet.ADJ or w_p == wordnet.VERB:
        return True
    else:
        return False



def read_our_dataset(concept_num=3, total_concept_num=10):
    with open('./train_tokens.p', 'rb') as f1:
        data_tra = pickle.load(f1)
    with open('./valid_tokens.p', 'rb') as f2:
        data_val = pickle.load(f2)
    with open('./test_tokens.p', 'rb') as f3:
        data_tst = pickle.load(f3)
    with open('./vocab_preproc.p', 'rb') as f4:
        vocab = pickle.load(f4)
    word2index = vocab.word2index
    word2count = vocab.word2count
    index2word = vocab.index2word
    n_words= vocab.n_words

    VAD = json.load(open("VAD.json", "r", encoding="utf-8"))
    concept = json.load(open("ConceptNet_VAD_dict1.json", "r", encoding="utf-8"))
    #在原数据的基础上添加新的字典项
    data_tra['concepts'], data_val['concepts'], data_tst['concepts'] = [], [], []
    data_tra['sample_concepts'], data_val['sample_concepts'], data_tst['sample_concepts'] = [], [], []
    data_tra['vads'], data_val['vads'], data_tst['vads'] = [], [], []  # each sentence's vad vectors
    data_tra['vad'], data_val['vad'], data_tst['vad'] = [], [], []  # each word's emotion intensity
    data_tra['target_vad'], data_val['target_vad'], data_tst['target_vad'] = [], [], []  # each target word's emotion intensity
    data_tra['target_vads'], data_val['target_vads'], data_tst['target_vads'] = [], [], []  # each target word's vad vectors

    # train
    train_contexts = data_tra['dialogue']  #三维List
    for i, sample in enumerate(train_contexts):  # sample:[[]] 二维list
        vads = []  # each item is sentence, each sentence contains a list word' vad vectors
        vad = []
        concepts = []  # concepts of each sample
        total_concepts = []
        total_concepts_tid = []
        for j, sentence in enumerate(sample):  # sentence:每一个单轮对话 即一维[]
            words_pos = nltk.pos_tag(sentence)  # 词性标注
            # 把每一个word的VAD向量存进来
            vads.append([VAD[word] if word in word2index and word in VAD else [0.5, 0.0, 0.5] for word in sentence])
            vad.append([emotion_intensity(VAD, word) if word in VAD else 0.0 for word in sentence])
            # vads，vad，sentence_concepts的长度和输入的文本一致
            sentence_concepts = [  # 为每一个单词，把他的相关concept拿过来[[第一个相关概念]，[第二个相关概念]...]
                concept[word] if word in word2index and word not in stop_words and word in concept and wordCate(
                    words_pos[wi]) else []
                for wi, word in enumerate(sentence)]

            sentence_concept_words = []  # for each sentence
            sentence_concept_vads = []
            sentence_concept_vad = []

            for cti, uc in enumerate(
                    sentence_concepts):  # sentence_concepts和输入的对话长度一致 # filter concepts of each token, complete their VAD value, select top total_concept_num.
                concept_words = []  # 把相关性小的概念去掉
                concept_vads = []
                concept_vad = []
                if uc != []:  # 当前对话中的某个概念所对的所有concept_words
                    for c in uc:  # iterate the concept lists [c,r,w] of each token
                        if c[1] not in REMOVE_RELATIONS and c[0] not in stop_words and c[0] in word2index:  # remove concpets that are stopwords or not in the dict
                            if c[0] in VAD and emotion_intensity(VAD, c[0]) >= 0.6:
                                concept_words.append(c[0])
                                concept_vads.append(VAD[c[0]])
                                concept_vad.append(emotion_intensity(VAD, c[0]))
                                total_concepts.append(c[0])  # all concepts of a sentence
                                total_concepts_tid.append([j, cti])  # the token that each concept belongs to

                    # concept_words = concept_words[:5]
                    # concept_vads = concept_vads[:5]
                    # concept_vad = concept_vad[:5]
                    concept_words = concept_words[:concept_num]  # []
                    concept_vads = concept_vads[:concept_num]  # [ [] ,[] ]
                    concept_vad = concept_vad[:concept_num]  # []   #只保留几个关系较大的

                sentence_concept_words.append(concept_words)  # 二维list,[[],[相关概念]..]
                sentence_concept_vads.append(concept_vads)
                sentence_concept_vad.append(concept_vad)

            sentence_concepts = [sentence_concept_words, sentence_concept_vads, sentence_concept_vad]  # 存放三个得分
            concepts.append(sentence_concepts)
        data_tra['concepts'].append(concepts)  # 三维列表，分别存放三种元素 见371行
        data_tra['sample_concepts'].append([total_concepts, total_concepts_tid])
        data_tra['vads'].append(vads)
        data_tra['vad'].append(vad)

    train_targets = data_tra['response']
    for i, target in enumerate(train_targets):
        # each item is the VAD info list of each target token
        data_tra['target_vads'].append(
            [VAD[word] if word in word2index and word in VAD else [0.5, 0.0, 0.5] for word in target])
        data_tra['target_vad'].append(
            [emotion_intensity(VAD, word) if word in VAD and word in word2index else 0.0 for word in target])
    print("trainset finish.")

    # valid
    valid_contexts = data_val['dialogue']
    for i, sample in enumerate(valid_contexts):
        vads = []  # each item is sentence, each sentence contains a list word' vad vectors
        vad = []
        concepts = []
        total_concepts = []
        total_concepts_tid = []

        for j, sentence in enumerate(sample):
            words_pos = nltk.pos_tag(sentence)

            vads.append(
                [VAD[word] if word in word2index and word in VAD else [0.5, 0.0, 0.5] for word in sentence])
            vad.append([emotion_intensity(VAD, word) if word in VAD else 0.0 for word in sentence])

            sentence_concepts = [
                concept[word] if word in word2index and word not in stop_words and word in concept and wordCate(
                    words_pos[wi]) else []
                for wi, word in enumerate(sentence)]

            sentence_concept_words = []  # for each sentence
            sentence_concept_vads = []
            sentence_concept_vad = []

            for cti, uc in enumerate(sentence_concepts):
                concept_words = []  # for each token
                concept_vads = []
                concept_vad = []
                if uc != []:
                    for c in uc:
                        if c[1] not in REMOVE_RELATIONS and c[0] not in stop_words and c[0] in word2index:
                            if c[0] in VAD and emotion_intensity(VAD, c[0]) >= 0.6:
                                concept_words.append(c[0])
                                concept_vads.append(VAD[c[0]])
                                concept_vad.append(emotion_intensity(VAD, c[0]))

                                total_concepts.append(c[0])
                                total_concepts_tid.append([j, cti])

                    concept_words = concept_words[:concept_num]
                    concept_vads = concept_vads[:concept_num]
                    concept_vad = concept_vad[:concept_num]

                sentence_concept_words.append(concept_words)
                sentence_concept_vads.append(concept_vads)
                sentence_concept_vad.append(concept_vad)

            sentence_concepts = [sentence_concept_words, sentence_concept_vads, sentence_concept_vad]
            concepts.append(sentence_concepts)

        data_val['concepts'].append(concepts)
        data_tra['sample_concepts'].append([total_concepts, total_concepts_tid])
        data_val['vads'].append(vads)
        data_val['vad'].append(vad)

    valid_targets = data_val['response']
    for i, target in enumerate(valid_targets):
        data_val['target_vads'].append(
            [VAD[word] if word in word2index and word in VAD else [0.5, 0.0, 0.5] for word in target])
        data_val['target_vad'].append(
            [emotion_intensity(VAD, word) if word in VAD and word in word2index else 0.0 for word in target])
    print('validset finish.')

    # test
    test_contexts = data_tst['dialogue']
    for i, sample in enumerate(test_contexts):
        vads = []  # each item is sentence, each sentence contains a list word' vad vectors
        vad = []
        concepts = []
        total_concepts = []
        total_concepts_tid = []
        for j, sentence in enumerate(sample):
            words_pos = nltk.pos_tag(sentence)

            vads.append(
                [VAD[word] if word in word2index and word in VAD else [0.5, 0.0, 0.5] for word in sentence])
            vad.append([emotion_intensity(VAD, word) if word in VAD else 0.0 for word in sentence])

            sentence_concepts = [
                concept[
                    word] if word in word2index and word not in stop_words and word in concept and wordCate(
                    words_pos[wi]) else []
                for wi, word in enumerate(sentence)]

            sentence_concept_words = []  # for each sentence
            sentence_concept_vads = []
            sentence_concept_vad = []

            for cti, uc in enumerate(sentence_concepts):
                concept_words = []  # for each token
                concept_vads = []
                concept_vad = []
                if uc != []:
                    for c in uc:
                        if c[1] not in REMOVE_RELATIONS and c[0] not in stop_words and c[0] in word2index:
                            if c[0] in VAD and emotion_intensity(VAD, c[0]) >= 0.6:
                                concept_words.append(c[0])
                                concept_vads.append(VAD[c[0]])
                                concept_vad.append(emotion_intensity(VAD, c[0]))

                                total_concepts.append(c[0])
                                total_concepts_tid.append([j, cti])

                    concept_words = concept_words[:concept_num]
                    concept_vads = concept_vads[:concept_num]
                    concept_vad = concept_vad[:concept_num]

                sentence_concept_words.append(concept_words)
                sentence_concept_vads.append(concept_vads)
                sentence_concept_vad.append(concept_vad)

            sentence_concepts = [sentence_concept_words, sentence_concept_vads, sentence_concept_vad]
            concepts.append(sentence_concepts)

        data_tst['concepts'].append(concepts)
        data_tra['sample_concepts'].append([total_concepts, total_concepts_tid])
        data_tst['vads'].append(vads)
        data_tst['vad'].append(vad)

    test_targets = data_tst['response']
    for i, target in enumerate(test_targets):
        data_tst['target_vads'].append(
            [VAD[word] if word in word2index and word in VAD else [0.5, 0.0, 0.5] for word in target])
        data_tst['target_vad'].append(
            [emotion_intensity(VAD, word) if word in VAD and word in word2index else 0.0 for word in target])
    print('testset finish.')

    return data_tra, data_val, data_tst, vocab





if __name__ == '__main__':

    # step 1
    get_concept_dict()

    # step 2
    rank_concept_dict()

    # step 3 save data
    # data_tra, data_val, data_tst, vocab = read_our_dataset()

    # with open('./cache_dialogue_2dim/kemp_train_new.p', "wb") as f:
    #     pickle.dump(data_tra, f)
    #     print("Saved kemp_dataset_train.p")
    # with open('./cache_dialogue_2dim/kemp_valid_new.p', "wb") as f1:
    #     pickle.dump(data_val, f1)
    #     print("Saved kemp_dataset_valid.p")
    # with open('./cache_dialogue_2dim/kemp_test_new.p', "wb") as f2:
    #     pickle.dump(data_tst, f2)
    #     print("Saved kemp_dataset_test.p")

    with open('./kemp_train_new.p', "rb") as ff:
        data_train = pickle.load(ff)
    with open('./kemp_valid_new.p', "rb") as ff1:
        data_valid = pickle.load(ff1)
    with open('./kemp_test_new.p', "rb") as ff2:
        data_test = pickle.load(ff2)
    print("===")