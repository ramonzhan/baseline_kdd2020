# coding: utf-8
import pickle as pkl
import fasttext as ft
import numpy as np
import os
import argparse
from sklearn import metrics
from metrics.metrics import Coverage, OneError
from multiprocessing import cpu_count


parser = argparse.ArgumentParser(description='preprocess.py')
parser.add_argument('-data_dir', type=str, default="/kdd2020/dataset/fasttext/it_1/")
parser.add_argument('-lr', type=float, default=1e-3)
parser.add_argument('-epoch', type=int, default=20)
parser.add_argument('-dim', type=int, default=300)
parser.add_argument('-prefix', type=str, default="_label_")
parser.add_argument('-k', type=int, default=20)
opt = parser.parse_args()


def make_ndarray(label_list, prob_array, skill_num):
    batch_size = len(label_list)
    # make pred
    pred = np.zeros([batch_size, skill_num])
    row_indice = [i for i in range(batch_size) for _ in label_list[i]]
    col_indice = [j for i in range(batch_size) for j in label_list[i]]
    pred[row_indice, col_indice] = 1
    # make confidence
    confi = np.zeros([batch_size, skill_num])
    last_row = 0
    col_prob = 0
    for row_idx, col_idx in zip(row_indice, col_indice):
        if row_idx == last_row:
            col_prob += 1
        else:
            col_prob = 1
            last_row = row_idx
        confi[row_idx, col_idx] = prob_array[row_idx, col_prob - 1]
    return pred, confi


def eval(cls, x, target, dim, k=20):
    labels, prob = cls.predict(x, k=k)
    labels = [[int(label[7 : ]) for label in row] for row in labels]
    predict, confidence = make_ndarray(labels, prob, dim)
    print("evaluating .... ")
    micro_f1 = metrics.f1_score(target, predict, average="micro")  #
    micro_p = metrics.precision_score(target, predict, average="micro")
    micro_r = metrics.recall_score(target, predict, average="micro")
    micro_auc = metrics.roc_auc_score(target, confidence, average="micro")
    hamming_loss = metrics.hamming_loss(target, predict)
    ranking_loss = metrics.label_ranking_loss(target, confidence)
    coverage = Coverage(confidence, target)
    oneerror = OneError(confidence, target)
    print("F1: [{}], P: [{}], R: [{}], AUC: [{}], \n hloss: [{}], rloss: [{}] cov: [{}], oerror: [{}] \t"
          .format(micro_f1, micro_p, micro_r, micro_auc,
                  hamming_loss, ranking_loss, coverage, oneerror,
                  ))


def main():
    train_file_dir = os.path.join(opt.data_dir, "train.txt")
    test_dir = os.path.join(opt.data_dir, "test.pkl")
    file_content, labels_list, label_dim = pkl.load(open(test_dir, "rb"))
    classifier = ft.train_supervised(train_file_dir,
                                     label_prefix=opt.prefix,
                                     epoch=opt.epoch,
                                     lr=opt.lr,
                                     dim=opt.dim,
                                     verbose=2,
                                     thread=cpu_count())

    eval(classifier, file_content, labels_list, label_dim, opt.k)


if __name__ == '__main__':
    main()

