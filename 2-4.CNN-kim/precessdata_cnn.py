import pickle as pkl
import numpy as np
import torch

#直接利用了SGM预处理的数据
def readdata():
    data = {}

    #train
    trainsrc = open("/kdd2020/dataset/sgm/it_1/trainsrc.str",'r')
    traintgt = open("/kdd2020/dataset/sgm/it_1/traintgt.str",'r')

    trainsrc = trainsrc.readlines()
    train_src = []
    for line in trainsrc:
        linelist = list(line.strip().split(" "))
        train_src.append(linelist)
    data["train_x"] = train_src

    traintgt = traintgt.readlines()
    train_tgt = []
    for line in traintgt:
        linelist = list(map(int, line.strip().split(" ")))
        train_tgt.append(linelist)
    data["train_y"] = train_tgt

    #test
    testsrc = open("/kdd2020/dataset/sgm/it_1/testsrc.str", 'r')
    testtgt = open("/kdd2020/dataset/sgm/it_1/testtgt.str", 'r')

    testsrc = testsrc.readlines()
    test_src = []
    for line in testsrc:
        linelist = list(line.strip().split(" "))
        test_src.append(linelist)
    data["test_x"] = test_src
    data["dev_x"] = test_src

    testtgt = testtgt.readlines()
    test_tgt = []
    for line in testtgt:
        linelist = list(map(int, line.strip().split(" ")))
        test_tgt.append(linelist)
    data["test_y"] = test_tgt
    data["dev_y"] = test_tgt

    #skill vocab
    skill_class = pkl.load(open("/kdd2020/dataset/it_1/skills.pkl",'rb'))
    a = set([item for sublist in skill_class.values() for item in sublist])
    data["classes"] = list(a)

    vocab = open("/kdd2020/dataset/sgm/it_1/src.dict", 'r')
    vocab = vocab.readlines()
    word_to_idx = {}
    idx_to_word = {}
    for word in vocab:
        wordlist = word.strip().split(" ")
        if wordlist[1] == '':
            wordlist[1] = 1
        word_to_idx[wordlist[0]] = int(wordlist[1])
        idx_to_word[int(wordlist[1])] = wordlist[0]
    vocab_list = list(word_to_idx.keys())
    data['vocab'] = vocab_list
    data['word_to_idx'] = word_to_idx
    data['idx_to_word'] = idx_to_word
    return data

def multi_hot(label_list, skill_num, batch_size, gpu):
    label_t = torch.zeros(batch_size, skill_num, requires_grad=False).to(gpu)
    row_indice = [i for i in range(batch_size) for _ in label_list[i]]
    col_indice = [-j - 1 for i in range(batch_size) for j in label_list[i]]
    label_t[row_indice, col_indice] = 1
    return label_t


if __name__ == "__main__":
    readdata()