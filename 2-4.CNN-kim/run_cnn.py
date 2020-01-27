# coding : utf-8
from model import CNN
from torch.autograd import Variable
import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.utils import shuffle
import argparse
from precessdata import readdata, multi_hot
from sklearn import metrics
from metrics.metrics import Coverage, OneError
import numpy as np


sigmoid = nn.Sigmoid()


def train(data, params):
    model = CNN(**params).cuda(params["GPU"])
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adadelta(parameters, params["LEARNING_RATE"])
    #改成多标签分类
    criterion = nn.BCEWithLogitsLoss(reduction="sum")
    max_f1 = 0
    for e in range(params["EPOCH"]):
        data["train_x"], data["train_y"] = shuffle(data["train_x"], data["train_y"])
        for i in range(0, len(data["train_x"]), params["BATCH_SIZE"]):
            batch_range = min(params["BATCH_SIZE"], len(data["train_x"]) - i)
            batch_x = [[data["word_to_idx"][w] for w in sent] +
                       [params["VOCAB_SIZE"] + 1] * (params["MAX_SENT_LEN"] - len(sent))
                       for sent in data["train_x"][i:i + batch_range]]

            y_labels = data["train_y"][i:i + batch_range]
            batch_y = multi_hot(y_labels, len(data["classes"]), batch_range, params["GPU"])
            batch_x = Variable(torch.LongTensor(batch_x)).cuda(params["GPU"])

            optimizer.zero_grad()
            model.train()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(parameters, max_norm=params["NORM_LIMIT"])
            optimizer.step()

        micro_f1, micro_p, micro_r, micro_auc, hamming_loss, ranking_loss, coverage, oneerror = \
            eval(data, model, params)

        print("E:[{}], F1: [{}], P: [{}], R: [{}], AUC: [{}], \n hloss: [{}], rloss: [{}] cov: [{}], oerror: [{}] \t"
              .format(e + 1,
                      micro_f1, micro_p, micro_r, micro_auc,
                      hamming_loss, ranking_loss, coverage, oneerror,
                      ))
        if micro_f1 > max_f1:
            max_f1 = micro_f1
    print("best test f1:", max_f1)


def eval(data, model, params):
    model.eval()
    x, y = data["test_x"], data["test_y"]
    x = [[data["word_to_idx"][w] if w in data["vocab"] else params["VOCAB_SIZE"] for w in sent] +
         [params["VOCAB_SIZE"] + 1] * (params["MAX_SENT_LEN"] - len(sent))
         for sent in x]

    x = Variable(torch.LongTensor(x)).cuda(params["GPU"])
    target = multi_hot(y, len(data["classes"]), len(y), params["GPU"]).cpu().data.numpy()
    confidence = sigmoid(model(x)).detach().cpu().data.numpy()
    predict = np.array(list(map(lambda x: list(map(lambda y: 1.0 if y > 0.5 else 0.0, x)), confidence)))

    print("evaluating .... ")
    micro_f1 = metrics.f1_score(target, predict, average="micro")  #
    micro_p = metrics.precision_score(target, predict, average="micro")
    micro_r = metrics.recall_score(target, predict, average="micro")
    micro_auc = metrics.roc_auc_score(target, confidence, average="micro")
    hamming_loss = metrics.hamming_loss(target, predict)
    ranking_loss = metrics.label_ranking_loss(target, confidence)
    coverage = Coverage(confidence, target)
    oneerror = OneError(confidence, target)
    return micro_f1, micro_p, micro_r, micro_auc, hamming_loss, ranking_loss, coverage, oneerror


def main():
    parser = argparse.ArgumentParser(description="-----[CNN-classifier]-----")
    parser.add_argument("--model", default="rand", help="available models: rand, static, non-static, multichannel")
    parser.add_argument("--dataset", default="it_1", help="available datasets: MR, TREC")
    parser.add_argument("--epoch", default=100, type=int, help="number of max epoch")
    parser.add_argument("--learning_rate", default=0.1, type=float, help="learning rate")
    parser.add_argument("--gpu", default=1, type=int, help="the number of gpu to be used")
    options = parser.parse_args()
    data = readdata()

    params = {
        "MODEL": options.model,
        "DATASET": options.dataset,
        "EPOCH": options.epoch,
        "LEARNING_RATE": options.learning_rate,
        "MAX_SENT_LEN": max([len(sent) for sent in data["train_x"] + data["dev_x"] + data["test_x"]]),
        "BATCH_SIZE": 32,
        "WORD_DIM": 128,
        "VOCAB_SIZE": len(data["vocab"]),
        "CLASS_SIZE": len(data["classes"]),
        "FILTERS": [3, 4, 5],
        "FILTER_NUM": [100, 100, 100],
        "DROPOUT_PROB": 0.5,
        "NORM_LIMIT": 3,
        "GPU": options.gpu
    }

    print("MODEL:", params["MODEL"])
    print("DATASET:", params["DATASET"])
    print("VOCAB_SIZE:", params["VOCAB_SIZE"])
    print("EPOCH:", params["EPOCH"])
    print("LEARNING_RATE:", params["LEARNING_RATE"])

    print("=" * 20 + "TRAINING STARTED" + "=" * 20)
    train(data, params)
    print("=" * 20 + "TRAINING FINISHED" + "=" * 20)


if __name__ == "__main__":
    main()
