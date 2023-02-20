import numpy as np
import re
import math


def conver_float(x):
    '''将词向量数据类型转换成可以计算的浮点类型'''
    float_str = x
    return [float(f) for f in float_str]


def process_wordembe(path):
    '''
    将词向量文件中的所有词向量存放到一个列表lines里
    :param path: a path of english word embbeding file 'glove.840B.300d.txt'
    :return: a list, element is a 301 dimension word embbeding, it's form like this
            ['- 0.12332 ... -0.34542\n', ', 0.23421 ... -0.456733/n', ..., 'you 0.34521 0.78905 ... -0.23123/n']
    '''
    f = open(path, 'r', encoding='utf-8')
    embed_lines = f.readlines()
    return embed_lines


def word2vec(x, lines):
    '''
    将一个字符串(这里指句子）中所有的词都向量化，并存放到一个列表里
    :param x: a sentence/sequence, type is string, for example 'hello, how are you ?'
    :return: a list, the form like [[word_vector1],...,[word_vectorn]], save per word embbeding of a sentence.
    '''
    x = x.split()[:-1]
    x_words = []
    for w in x:
        for line in lines:
            # print(line)
            if w == line.split()[0]:  # 将词向量按空格切分到一个列表里，将列表的第一个词与x的word比较
                print(w)
                x_words.append(conver_float(line[:-1].split()[1:]))  # 若在词向量列表中找到对应的词向量，添加到x_words列表里
                break
    return x_words


def sentence_embedding(x_words):
    '''
    上面的第一个公式：computing sentence embedding by computing average of all word embeddings of sentence.
    :param x: a sentence, type is string.
    :param x_words: list[list1, list2,...,listn], listk(k=1...n) is word vector which from sentence x,
    :param y_words: list[list1, list2,..., listn], listk(k=1...n) is word vector which from sentence y,
    :return: a scalar, it's value is in [0, 1]
    '''
    sen_embed = np.array([0 for _ in range(len(x_words[0]))])  # 存放句向量
    print(len(sen_embed))

    for x_v in x_words:
        x_v = np.array(x_v)
        print(len(x_v))
        sen_embed = np.add(x_v, sen_embed)
    sen_embed = sen_embed / math.sqrt(sum(np.square(sen_embed)))
    return sen_embed


def cosine_similarity(x, y, norm=False):
    """ 向量均值法EA:计算两个向量x和y的余弦相似度 """
    assert len(x) == len(y), "len(x) != len(y)"
    zero_list = np.array([0 for _ in range(len(x))])
    print(zero_list)
    if x.all() == zero_list.all() or y.all() == zero_list.all():
        return float(1) if x == y else float(0)

    # method 1
    res = np.array([[x[i] * y[i], x[i] * x[i], y[i] * y[i]] for i in range(len(x))])
    cos = sum(res[:, 0]) / (np.sqrt(sum(res[:, 1])) * np.sqrt(sum(res[:, 2])))

    return 0.5 * cos + 0.5 if norm else cos  # 归一化到[0, 1]区间内


if __name__ == '__main__':
    # print(cosine_similarity([1, 1], [0, 0]))   # 0.0
    # print(cosine_similarity([1, 1], [-1, -1]))  # -1.0
    # print(cosine_similarity([1, 1], [2, 2]))  # 1.0
    # f = open('G:\\PycharmProjects\\test\\glove.840B.300d.txt', 'r', encoding='utf-8')
    path = 'G:\\PycharmProjects\\test\\glove.840B.300d.txt'

    embed_lines = process_wordembe(path)
    # print(lines[:1])
    x = "what 's wrong ? \n"
    y = "I 'm fine . \n"
    x_words = word2vec(x, embed_lines)
    y_words = word2vec(y, embed_lines)

    # print(x_words)
    # print(y_words)

    # 举例
    # x_words = [[1.0, 2.0], [3.0, 4.0], [4, 5]]
    x_emb = sentence_embedding(x_words)
    y_emb = sentence_embedding(y_words)
    # print(x_emb.shape)   #(300,)
    # print(y_emb.shape)   #(300,)

    embedding_average = cosine_similarity(x_emb, y_emb)
    print(embedding_average)