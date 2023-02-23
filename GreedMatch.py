import numpy as np
import re


def cosine_similarity(x, y, norm=False):
    """ Computes the cosine similarity of two vectors x and y """
    assert len(x) == len(y), "len(x) != len(y)"
    zero_list = [0] * len(x)
    if x == zero_list or y == zero_list:
        return float(1) if x == y else float(0)

    # method 1
    res = np.array([[x[i] * y[i], x[i] * x[i], y[i] * y[i]] for i in range(len(x))])
    cos = sum(res[:, 0]) / (np.sqrt(sum(res[:, 1])) * np.sqrt(sum(res[:, 2])))

    return 0.5 * cos + 0.5 if norm else cos  # Normalized to [0, 1] interval


def conver_float(x):
    '''Convert the word vector data type to a floating point type that can be calculated'''
    float_str = x
    return [float(f) for f in float_str]


def process_wordembe(path):
    '''
    Store all word vectors in the word vector file into a list lines
    :param path: a path of english word embbeding file 'glove.840B.300d.txt'
    :return: a list, element is a 301 dimension word embbeding, it's form like this
            ['- 0.12332 ... -0.34542\n', ', 0.23421 ... -0.456733\n', ..., 'you 0.34521 0.78905 ... -0.23123\n']
    '''
    f = open(path, 'r', encoding='utf-8')
    lines = f.readlines()
    return lines


def word2vec(x, lines):
    '''
    Vectorize all the words in a string (a sentence in the study) and store them in a list
    :param x: a sentence/sequence, type is string, for example 'hello, how are you ?'
    :return: a list, the form like [[word_vector1],...,[word_vectorn]], save per word embbeding of a sentence.
    '''
    x = x.split()[:-1]
    x_words = []
    for w in x:
        for line in lines:
            # print(line)
            if w == line.split()[0]:  # Split the word vector into a list by spaces, and compare the first word of the list with the word of x
                print(w)
                x_words.append(conver_float(line[:-1].split()[1:]))  # If the corresponding word vector is found in the word vector list, add it to the x_words list
                break
    return x_words


def greedy(x, x_words, y_words):
    '''
    The first equation mentioned above
    :param x: a sentence, type is string.
    :param x_words: list[list1, list2,...,listn], listk(k=1...n) is word vector which from sentence x,
    :param y_words: list[list1, list2,..., listn], listk(k=1...n) is word vector which from sentence y,
    :return: a scalar, it's value is in [0, 1]
    '''
    cosine = []  # Stores the cosine similarity between a word in one sentence and all words in another sentence
    sum_x = 0  # store the final result
    for x_v in x_words:
        for y_v in y_words:
            cosine.append(cosine_similarity(x_v, y_v))
        if cosine:
            sum_x += max(cosine)
            cosine = []
    sum_x = sum_x / len(x.split()[:-1])
    return sum_x


def greedy_match(path, x, y):
    '''
    The second equation above
    :param lines: english word embbeding list, like[['-','0.345',...,'0.3123'],...]
    :param x: a sentence, here is a candidate answer
    :param y: a sentence, here is reference answer
    :return: a scalar in [0,1]
    '''
    lines = process_wordembe(path)
    # x_words.append(line.split()[1:] for line in lines for w in x if w in line)
    x_words = word2vec(x, lines)
    y_words = word2vec(y, lines)

    # greedy match
    sum_x = greedy(x, x_words, y_words)
    sum_y = greedy(y, y_words, x_words)
    score = (sum_x + sum_y) / 2
    return score


if __name__ == '__main__':
    # print(cosine_similarity([1, 1], [0, 0]))   # 0.0
    # print(cosine_similarity([1, 1], [-1, -1]))  # -1.0
    # print(cosine_similarity([1, 1], [2, 2]))  # 1.0
    f = open('G:\\PycharmProjects\\test\\glove.840B.300d.txt', 'r', encoding='utf-8')  # Change here to the path of your own project
    path = 'G:\\PycharmProjects\\test\\glove.840B.300d.txt'  # Change here to the path of your own project

    # lines = process_wordembe(path)
    # print(lines[:1])

    x = "发生什么事了? \n"
    y = "我很好 \n"
    # x_words = word2vec(x, lines)
    # y_words = word2vec(y, lines)
    # print(x_words[0])
    # print(y_words[0])
    # sum = greedy(x, x_words, y_words)
    # print(sum)

    score = greedy_match(path, x, y)
    print(score)



