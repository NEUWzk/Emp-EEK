import pandas as pd
import csv
from torch.utils.data import Dataset, DataLoader
import hashlib
import os
import pickle
from tqdm import tqdm

class RegressionDataset(Dataset):
    def __init__(self, filename):
        x = pd.read_csv(filename)
        self.text = list(x["utterance"]) #通过pandas加载的一列数据，类型为Serise,可以通过list强制转化为列表
        self.labels = list(x["sentiment"])  #一列数据，可以通过 a[index]来调用具体的
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.text[index], self.labels[index] #([text],[score])
    
    def collate_fn(self, data):
        dat = pd.DataFrame(data) #传进来一个tuple
        return [dat[i].tolist() for i in dat] #返回一个二维列表，list[i]表示每个batch地数据
    
def RegressionLoader(filename, batch_size, shuffle):
    dataset = RegressionDataset(filename)
    loader = DataLoader(dataset, shuffle=shuffle, batch_size=batch_size, collate_fn=dataset.collate_fn)
    return loader

class ClassificationDataset(Dataset):
    def __init__(self, filename):
        
        x = pd.read_csv(filename)
        self.context = list(x["seeker_post"])
        self.response = list(x["response_post"])
        self.labels = list(x["label"])
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.context[index], self.response[index], self.labels[index]
    
    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [dat[i].tolist() for i in dat]
    
def ClassificationLoader(filename, batch_size, shuffle):
    dataset = ClassificationDataset(filename)
    loader = DataLoader(dataset, shuffle=shuffle, batch_size=batch_size, collate_fn=dataset.collate_fn)
    return loader

class EmpatheticDialogues(Dataset): #和前面两个项目不同，是在Dataset里面处理数据集的，用于构建单轮、多轮对话及其回复
    def __init__(self, filename, cache="./.cache"):
        if cache:
            cs = hashlib.md5(open(filename, "rb").read()).hexdigest()
            cache_file = f"{cache}/{cs}"
            if os.path.isfile(cache_file):
                with open(cache_file, "rb") as f:
                    self.data = pickle.load(f)
                print(f"Loaded data from {filename} cache")
                return
        data = pd.read_csv(filename, quoting=0).drop(columns=["history"])
        #train_dpr.csv:(84169,15),和桌面的ED数据集是一致的
        conversations = data["conv_id"].unique().tolist() #unique方法用来去重，把conv_id一样的合并
        #经过这一步，已经将所有不一样的 conv_id 放进了一个列表，每一维是str类型
        self.data = [] #data应该放在所有循环的最外面，用于存放整个对话、回复数据集（字典形式）
        print(f"Loading data from {filename}")
        for conv_id in tqdm(conversations): #把conversations当作索引
            conv = data.query(f'conv_id == "{conv_id}"').sort_values("utterance_idx") #conv是一个表格，行数为conv_id == "{conv_id}满足这样的
            context = [] #中间容器
            for idx, utterance in enumerate(conv.iterrows()): #遍历表格的每一行，把一行当作一个对象
                #utterance是一个tuple，第一维是索引用于记录行数，第二维才是真实的数据
                utterance = utterance[1]
                curr_utterance = utterance["utterance"].replace("_comma_", ",") #拿到对话文本句子，str
                if idx % 2 == 1: #对于奇数轮次（索引为奇数）的对话，将本轮对话作为回复，前面的当成对话上下文
                    self.data.append({
                        "conv_id": conv_id,
                        "emotion": utterance["context"],
                        "context": context[:idx+1],
                        "response": curr_utterance, #把当前轮次的句子作为回复
                        "exemplars": utterance["exemplars_empd_reddit"].split("ææ"),
                        "empathy1_labels": int(utterance["empathy_labels"]),
                        "empathy2_labels": int(utterance["empathy2_labels"]),
                        "empathy3_labels": int(utterance["empathy3_labels"]),
                        "sentiment": utterance["sentiment"]
                        })
                context.append(curr_utterance)
        # print(len(self.data)) #40254
        if cache:
            if not os.path.isdir(cache):
                os.mkdir(cache)
            with open(cache_file, "wb") as f:
                pickle.dump(self.data, f)

    def __len__(self):
        return len(self.data) #每轮对话都是一个类似字典的格式

    def __getitem__(self, index):
        return self.data[index]  #是一个字典

    def collate_fn(self, data): #输入进来的data是一个一维列表，每一项是个字典
        dat = pd.DataFrame(data)
        return [dat[i].tolist() for i in dat] #把一列放在一起，[[happy,sad],[1,3]],二维列表

def MainDataLoader(filename, batch_size, shuffle, cache="./.cache"):
    #构建DataLoader时，一是通过Dataset来实例化对象，二是通过DataLoader创建真正的 Loader
    dataset = EmpatheticDialogues(filename, cache=cache) #此处的EmpatheticDialogues等同于Dataset
    #已经将数据存放在了 .cache路径下，使用的时候直接pickle.load,不用每次再创建一遍数据集了
    loader = DataLoader(dataset, shuffle=shuffle, batch_size=batch_size, collate_fn=dataset.collate_fn)
    #DataLoader每次再取数据时，根据batch_size取几个索引，然后调用 __getitem__函数，取对应的数据，存到一个列表，最后把__getitem__函数之后的结果传递给collate_fn

    return loader
