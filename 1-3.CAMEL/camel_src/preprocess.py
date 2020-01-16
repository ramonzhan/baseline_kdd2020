# coding: utf-8
"""
1. 用gensim中的doc2vec学习文本表示，制作样本矩阵
2. 制作标签矩阵
"""
import os
import pickle as pkl
import numpy as np
import logging
from multiprocessing import cpu_count
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


def load_data(path):
    # dir
    train_test_dir = os.path.join(path, "train_test.pkl")
    jobdesc_dir = os.path.join(path, "jobdesc.pkl")
    cor_label_dir = os.path.join(path, "skills.pkl")
    scontent_dir = os.path.join(path, "skill_vocab.pkl")
    # load
    train_idxs, test_idxs = pkl.load(open(train_test_dir, "rb"))
    jobdesc_dict, max_jd_len, pad_idx = pkl.load(open(jobdesc_dir, "rb"))
    cor_label_dict = pkl.load(open(cor_label_dir, "rb"))
    _, __, skill_num = pkl.load(open(scontent_dir, "rb"))
    # split
    train_x, train_y, test_x, test_y = [], [], [], []
    for train_idx in train_idxs:
        train_x.append(jobdesc_dict[train_idx]), train_y.append(cor_label_dict[train_idx])
    for test_idx in test_idxs:
        test_x.append(jobdesc_dict[test_idx]), test_y.append(cor_label_dict[test_idx])
    return train_x, train_y, test_x, test_y, skill_num


def y_matrix(raw_y, label_num):
    """
    制作CAMEL需要的标签矩阵。ndarray, shape = 样本个数 x 标签个数。int类型
    :param raw_y: list的list，外层list的元素是每个样本的所有标签list，内层是一个个的标签（无序）
    :param label_num: 标签的个数
    :return: ndarray
    """
    instance_num = len(raw_y)
    matrix_y = np.zeros(shape=(instance_num, label_num), dtype=int)
    row_indice = [i for i in range(instance_num) for _ in raw_y[i]]
    col_indice = [-j - 1 for i in range(instance_num) for j in raw_y[i]]
    matrix_y[row_indice, col_indice] = 1
    return matrix_y


def x_matrix(raw_x, dim=256, epochs=50):
    x_train = []
    instance_num = len(raw_x)
    for i, token_list in enumerate(raw_x):
        doc = TaggedDocument(token_list, tags=[i])
        x_train.append(doc)
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
    model = Doc2Vec(x_train, vector_size=dim, workers=cpu_count(), window=5, epochs=epochs)
    matrix_x = np.zeros(shape=(instance_num, dim))
    for i in range(instance_num):
        docvec = model.docvecs[i]
        matrix_x[i] = docvec
    return matrix_x


def main():
    # dir
    data_path = "/home/rleating/kdd2020/dataset/toy"
    save_path = "/home/rleating/kdd2020/dataset/doc2vec"
    tr_x, tr_y, te_x, te_y, skill_num = load_data(data_path)
    # label
    mat_tr_y, mat_te_y = y_matrix(tr_y, skill_num), y_matrix(te_y, skill_num)    # convert to matrix
    # feature
    tr_x, te_x = [list(map(str, tlist)) for tlist in tr_x], [list(map(str, tlist)) for tlist in te_x]
    mat_tr_x, mat_te_x = x_matrix(tr_x), x_matrix(te_x)   # convert to matrix
    # save matrix
    te_x_dir = os.path.join(save_path, "test_x")
    te_y_dir = os.path.join(save_path, "test_y")
    tr_x_dir = os.path.join(save_path, "train_x")
    tr_y_dir = os.path.join(save_path, "train_y")
    np.save(te_x_dir, mat_te_x), np.save(te_y_dir, mat_te_y)
    np.save(tr_x_dir, mat_tr_x), np.save(tr_y_dir, mat_tr_y)
    print("训练特征形状 [{}] 测试特征形状 [{}] 训练标签形状[{}] 测试标签形状 [{}]"
          .format(mat_tr_x.shape, mat_te_x.shape, mat_tr_y.shape, mat_te_y.shape))


if __name__ == "__main__":
    main()
