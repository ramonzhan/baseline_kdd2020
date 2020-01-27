import numpy as np
import torch
import pickle

#列表集和元素总个数
def exclusive_combine(*in_list):
    res = set()
    in_list = list(*in_list)
    for n_l in in_list:
        for i in n_l:
            res.add(i)
    return list(res)


def multi_hot(label_list, skill_num, batch_size, gpu):
    label_t = torch.zeros(batch_size, skill_num, requires_grad=False).to(gpu)
    row_indice = [i for i in range(batch_size) for _ in label_list[i]]
    col_indice = [-j - 1 for i in range(batch_size) for j in label_list[i]]
    label_t[row_indice, col_indice] = 1
    return label_t

