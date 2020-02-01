# coding: utf-8
"""
两个用生成的方式来做多标签分类的模型的one-error评测
"""
# import numpy as np


def oneerror_gene(gene_seq, target):
    """
    根据生成的标签序列和目标标签来得到one-error的值
    :param gene_seq: list(list), 内层是每个instence的标签，外层是instence的list
    :param target: list(list)
    :return: one-error value
    """
    one_error = 0
    num = len(gene_seq)
    assert num == len(target)
    for idx, ins_pre in enumerate(gene_seq):
        if ins_pre[0] not in target[idx]:
            one_error += 1
    return one_error / num
