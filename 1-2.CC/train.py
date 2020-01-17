# coding: utf-8
import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import torch
import time
import numpy as np
import random
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


set_seed(214)


def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler="resolve")
    parser.add_argument("--doc2vec", help="if use doc2vec pretrained embedding", type=int, default=1)
    parser.add_argument("--svm", type=int, default=0)
    parser.add_argument("--root_dir", help="the root dir", type=str, default="../../")
    parser.add_argument("--dataset", help="the dir of all the used datasets", type=str, default="dataset")
    parser.add_argument("--results", help="the results dir", type=str, default="results")
    parser.add_argument("--data_num", help="the used dataset dir", type=str, default="it_1")
    args = parser.parse_args()
    return args


print("baseline experiment, cc")
def around(number, deci=3):
    return np.around(number, decimals=deci)


def main_svm(args):
    results_dir = os.path.join(args.root_dir, args.results)
    model_args = "baseline_svmcc"
    results_dir = os.path.join(results_dir, model_args)
    model_dir = os.path.join(args.root_dir, "models")
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    dataset_dir = os.path.join(args.root_dir, args.dataset, "doc2vec", args.data_num)
    trx_dir, try_dir = os.path.join(dataset_dir, "train_x.npy"), os.path.join(dataset_dir, "train_y.npy")
    tex_dir, tey_dir = os.path.join(dataset_dir, "test_x.npy"), os.path.join(dataset_dir, "test_y.npy")
    tr_x, tr_y, te_x, te_y = np.load(trx_dir), np.load(try_dir), np.load(tex_dir), np.load(tey_dir)
    label_dim = tr_y.shape[1]
    assert label_dim == te_y.shape[1]
    clf_list = []
    start = time.time()
    for i in range(label_dim):
        clf = LogisticRegression()
        clf.fit(tr_x, tr_y[:, i].ravel())
        pred = clf.predict(tr_x)
        tr_x = np.c_[tr_x, pred]
        clf_list.append(clf)
        minute = np.around((time.time() - start) / 60, decimals=4)
        print("已训练第[{}]个标签，用时[{}] min".format(i + 1, minute))

    joblib.dump(clf_list, model_dir + '/cc.pkl')


if __name__ == "__main__":
    args = parse_args()
    main_svm(args)
