# coding: utf-8
from skmultilearn.problem_transform import LabelPowerset
from sklearn.linear_model import LogisticRegression
import os
import numpy as np
from sklearn import metrics
from metrics.metrics import Coverage, OneError
from multiprocessing import cpu_count

root_dir = "/kdd2020/dataset"
data_num = "it_1"


dataset_dir = os.path.join(root_dir, "doc2vec", data_num)
trx_dir, try_dir = os.path.join(dataset_dir, "train_x.npy"), os.path.join(dataset_dir, "train_y.npy")
tex_dir, tey_dir = os.path.join(dataset_dir, "test_x.npy"), os.path.join(dataset_dir, "test_y.npy")
tr_x, tr_y, te_x, target = np.load(trx_dir), np.load(try_dir), np.load(tex_dir), np.load(tey_dir)
classifier = LabelPowerset(classifier=LogisticRegression(solver="liblinear", n_jobs=cpu_count()))
classifier.fit(tr_x, tr_y)
predict = classifier.predict(te_x).todense().A
confidence = classifier.predict_proba(te_x).todense().A
print("evaluating .... ")

micro_f1 = metrics.f1_score(target, predict, average="micro")  #
micro_p = metrics.precision_score(target, predict, average="micro")
micro_r = metrics.recall_score(target, predict, average="micro")
micro_auc = metrics.roc_auc_score(target, confidence, average="micro")

hamming_loss = metrics.hamming_loss(target, predict)
ranking_loss = metrics.label_ranking_loss(target, confidence)
coverage = Coverage(confidence, target)
oneerror = OneError(confidence, target)

print("LP model: \n micro_f1:[{}], micro_auc:[{}], p:[{}], r:[{}], \n"
      "hamming_loss:[{}], ranking_loss:[{}], cov:[{}], oneerror:[{}]"
      .format(micro_f1, micro_auc, micro_p, micro_r, hamming_loss, ranking_loss, coverage, oneerror))