# coding: utf-8
import os
import argparse
import pickle as pkl
import numpy as np


parser = argparse.ArgumentParser(description='preprocess.py')
parser.add_argument('-load_data', type=str, default="/kdd2020/dataset/it_1/")
parser.add_argument('-save_data', type=str, default="/kdd2020/dataset/fasttext/it_1/")
opt = parser.parse_args()


def traintest_split(src_dict, tgt_dict, train_idxs, test_idxs):
    """
    根据已有的train/test索引分别制作训练和测试用的dict（tgt和src）
    :param src_dict: 总的src dict key-jdidx，value对应的内容，
    :param tgt_dict: 总的tgt，key--jdidx，value对应的标签
    :param train_idxs: list，用于训练的jdidx
    :param test_idxs: list 用于测试的idx
    :return: train_src_dict, test_src_dict, train_tgt_dict, test_tgt_dict
    """
    train_src_dict, test_src_dict, train_tgt_dict, test_tgt_dict = dict(), dict(), dict(), dict()
    for train_idx in train_idxs:
        train_src_dict[train_idx] = src_dict[train_idx]
        train_tgt_dict[train_idx] = [-tgt - 1 for tgt in tgt_dict[train_idx]]
    for test_idx in test_idxs:
        test_src_dict[test_idx] = src_dict[test_idx]
        test_tgt_dict[test_idx] = [-tgt - 1 for tgt in tgt_dict[test_idx]]
    return train_src_dict, test_src_dict, train_tgt_dict, test_tgt_dict


def addprefix(labels):
    output_labels = []
    for label in labels:
        label = "_label_" + str(label)
        output_labels.append(label)
    return " ".join(output_labels)


def build_train(src, tgt, save_dir):
    file_content = []
    for idx in src.keys():
        jdcontent, labels = src[idx], tgt[idx]
        content = " ".join([str(token) for token in jdcontent])
        labels = addprefix(labels)
        row = labels + " " + content + "\n"
        file_content.append(row)
    with open(save_dir, "w") as f:
        f.writelines(file_content)


def multi_hot(label_list, skill_num):    # 这里不用乘负号再减一了
    batch_size = len(label_list)
    label_t = np.zeros([batch_size, skill_num])
    row_indice = [i for i in range(batch_size) for _ in label_list[i]]
    col_indice = [j for i in range(batch_size) for j in label_list[i]]
    label_t[row_indice, col_indice] = 1
    return label_t


def build_test(src, tgt, label_dim, save_dir):
    file_content, labels_list = [], []
    for idx in src.keys():
        jdcontent, labels = src[idx], tgt[idx]
        content = " ".join([str(token) for token in jdcontent])
        row = content
        file_content.append(row), labels_list.append(labels)
    labels_list = multi_hot(labels_list, label_dim)
    pkl.dump([file_content, labels_list, label_dim], open(save_dir, "wb"))


def main():
    if not os.path.exists(opt.save_data):
        os.mkdir(opt.save_data)
    # dir
    dataset_dir = os.path.join(opt.load_data, "train_test.pkl")
    jobdesc_dir = os.path.join(opt.load_data, "jobdesc.pkl")
    skills_dir = os.path.join(opt.load_data, "skills.pkl")
    skill_vocab_dir = os.path.join(opt.load_data, 'skill_vocab.pkl')
    save_train_dir = os.path.join(opt.save_data, "train.txt")   # 以_l_为prefix
    save_test_dir = os.path.join(opt.save_data, "test.pkl")    # test 保存成list的形式， 然后再保存一个ndarray的target进去

    # load data
    train_indexs, test_indexs = pkl.load(open(dataset_dir, "rb"))
    _, __, skill_num = pkl.load(open(skill_vocab_dir, "rb"))
    jdcontent, _, __ = pkl.load(open(jobdesc_dir, "rb"))
    skill_dict = pkl.load(open(skills_dir, "rb"))    # key: jd idx, value: 对应的技能idx列表

    print("split the train and test set...")
    train_src, test_src, train_tgt, test_tgt = traintest_split(jdcontent, skill_dict, train_indexs, test_indexs)

    print("Building train .txt file")
    build_train(train_src, train_tgt, save_train_dir)

    print("Build test file (pickle)")
    build_test(test_src, test_tgt, skill_num, save_test_dir)

if __name__ == "__main__":
    main()
