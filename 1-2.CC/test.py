# coding: utf-8
import numpy as np
from sklearn.externals import joblib
import os
from sklearn import metrics
from metrics.metrics import Coverage, OneError


def predict(x_test):
    root_dir = "/kdd2020"
    model_dir = os.path.join(root_dir, "models")
    clf_list = joblib.load(model_dir + '/cc.pkl')

    label_dim = len(clf_list)
    y = np.zeros((x_test.shape[0], label_dim))
    prob = np.zeros((x_test.shape[0], label_dim))

    for i in range(label_dim):
        y[:, i] = clf_list[i].predict(x_test)
        prob[:, i] = clf_list[i].predict_proba(x_test)[:, 1]
        x_test = np.c_[x_test, y[:, i]]
    return y, prob


def load_data(path):
    x_test = np.load(path + '/test_x.npy')
    y_test = np.load(path + '/test_y.npy')
    return x_test, y_test


if __name__ == '__main__':

    data_path = "/kdd2020/dataset/doc2vec/it_1"
    x_test, target = load_data(data_path)
    predict, confidence = predict(x_test)

    print("evaluating .... ")

    micro_f1 = metrics.f1_score(target, predict, average="micro")  #
    micro_p = metrics.precision_score(target, predict, average="micro")
    micro_r = metrics.recall_score(target, predict, average="micro")
    micro_auc = metrics.roc_auc_score(target, confidence, average="micro")

    hamming_loss = metrics.hamming_loss(target, predict)
    ranking_loss = metrics.label_ranking_loss(target, confidence)
    coverage = Coverage(confidence, target)
    oneerror = OneError(confidence, target)

    print("CC model: \n micro_f1:[{}], micro_auc:[{}], p:[{}], r:[{}], \n"
          "hamming_loss:[{}], ranking_loss:[{}], cov:[{}], oneerror:[{}]"
          .format(micro_f1, micro_auc, micro_p, micro_r, hamming_loss, ranking_loss, coverage, oneerror))