# coding: utf-8
import numpy as np
from sklearn.externals import joblib
import os
from sklearn import metrics


def predict(x_test):
    root_dir = "/home/rleating/kdd2020"
    model_dir = os.path.join(root_dir, "models")
    clf_list = joblib.load(model_dir + '/br.pkl')

    label_dim = len(clf_list)
    y = np.zeros((x_test.shape[0], label_dim))
    prob = np.zeros((x_test.shape[0], label_dim))

    for i in range(label_dim):
        y[:, i] = clf_list[i].predict(x_test)
        prob[:, i] = clf_list[i].predict_proba(x_test)[:, 1]
        return y, prob


def load_data(path):
    x_test = np.load(path + '/test_x.npy')
    y_test = np.load(path + '/test_y.npy')
    return x_test, y_test


if __name__ == '__main__':

    data_path = "/home/rleating/kdd2020/dataset/doc2vec/it_1"
    x_test, y_test = load_data(data_path)
    pred, pred_prob = predict(x_test)

    print("evaluating .... ")

    micro_f1 = metrics.f1_score(y_test, pred, average="micro")   #
    hamming_loss = metrics.hamming_loss(y_test, pred)
    ranking_loss = metrics.label_ranking_loss(y_test, pred_prob)
    micro_auc = metrics.roc_auc_score(y_test, pred_prob, average="micro")

    print("BR model: \n micro_f1:[{}], hamming_loss:[{}], ranking_loss:[{}], micro_auc:[{}]"
          .format(micro_f1, hamming_loss, ranking_loss, micro_auc))
