from nltk.corpus import wordnet
from nltk import pos_tag
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()


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

# 定义一个函数来对一个句子中的所有单词进行词形还原
def lemmatize_all(token):
    word, tag  = pos_tag([token])[0]
    if tag.startswith('NN'):
        return wnl.lemmatize(word, pos='n')
    elif tag.startswith('VB'):
        return wnl.lemmatize(word, pos='v')
    elif tag.startswith('JJ'):
        return wnl.lemmatize(word, pos='a')
    elif tag.startswith('R'):
        return wnl.lemmatize(word, pos='r')
    else:
        return wnl.lemmatize(word)


class Stack(object):
    # 初始化栈为空列表
    def __init__(self):
        self.items = []

    # 判断栈是否为空，返回布尔值
    def is_empty(self):
        return self.items == []

    # 返回栈顶元素
    def peek(self):
        return self.items[len(self.items) - 1]

    # 返回栈的大小
    def size(self):
        return len(self.items)

    # 把新的元素堆进栈里面（程序员喜欢把这个过程叫做压栈，入栈，进栈……）
    def push(self, item):
        self.items.append(item)

    # 把栈顶元素丢出去（程序员喜欢把这个过程叫做出栈……）
    def pop(self, item):
        return self.items.pop()

    def clean(self):
        while not self.is_empty():
            self.items.pop()
        return

