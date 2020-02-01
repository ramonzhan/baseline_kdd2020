# coding : utf-8
from torch.autograd import Variable
import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.utils import shuffle
from sklearn import metrics
import argparse
from precessdata import readdata, multi_hot
from cnn_rnn import CNN_RNN
from metrics.gene_oneerror import oneerror_gene


def train(data, params):
    model = CNN_RNN(**params).cuda(params["GPU"])
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adadelta(parameters, params["LEARNING_RATE"])
    #改成多标签分类
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.BCEWithLogitsLoss(reduction="sum")

    for e in range(params["EPOCH"]):
        data["train_x"], data["train_y"] = shuffle(data["train_x"], data["train_y"])
        trian_loss = 0
        for i in range(0, len(data["train_x"]), params["BATCH_SIZE"]):
            batch_range = min(params["BATCH_SIZE"], len(data["train_x"]) - i)

            batch_x = [[data["word_to_idx"][w] for w in sent] +
                       [params["VOCAB_SIZE"] + 1] * (params["MAX_SENT_LEN"] - len(sent))
                       for sent in data["train_x"][i:i + batch_range]]

            y_labels = data["train_y"][i:i + batch_range]

            batch_x = Variable(torch.LongTensor(batch_x)).cuda(params["GPU"])
            batch_y = torch.zeros((batch_range, 50), dtype=torch.long).cuda(params["GPU"])
            for idx, tgt in enumerate(y_labels):
                tgt_sl = len(tgt)
                batch_y[idx, :tgt_sl] = torch.LongTensor(tgt)

            optimizer.zero_grad()
            model.train()
            pred = model(batch_x, batch_y, 50, "train")
            loss = criterion(pred.view(-1, params["CLASS_SIZE"]), batch_y.view(-1))
            trian_loss += loss
            loss.backward()
            nn.utils.clip_grad_norm_(parameters, max_norm=params["NORM_LIMIT"])
            optimizer.step()

        micro_f1, micro_p, micro_r, hamming_loss, one_error = \
            eval(data, model, params)

        print("E:[{}], F1: [{}], P: [{}], R: [{}], hloss: [{}], one_erorr: [{}] \t"
              .format(e + 1,
                      micro_f1, micro_p, micro_r,
                      hamming_loss, one_error
                      ))


def eval(data, model, params):
    model.eval()
    x, y = data["test_x"], data["test_y"]
    x = [[data["word_to_idx"][w] if w in data["vocab"] else params["VOCAB_SIZE"] for w in sent] +
         [params["VOCAB_SIZE"] + 1] * (params["MAX_SENT_LEN"] - len(sent))
         for sent in x]

    x = Variable(torch.LongTensor(x)).cuda(params["GPU"])
    y = [[data["classes"].index(c) for c in inner] for inner in y]
    target = multi_hot(y, len(data["classes"]), len(y))

    batch_y = torch.zeros((len(y), 50), dtype=torch.long).cuda(params["GPU"])
    for idx, tgt in enumerate(y):
        tgt_sl = len(tgt)
        batch_y[idx, :tgt_sl] = torch.LongTensor(tgt)
        pred = model(x, batch_y, 50, "test").detach().cpu().numpy()
    predict = multi_hot(pred, len(data["classes"]), len(y))

    print("evaluating .... ")
    one_error = oneerror_gene(pred, y)
    micro_f1 = metrics.f1_score(target, predict, average="micro")  #
    micro_p = metrics.precision_score(target, predict, average="micro")
    micro_r = metrics.recall_score(target, predict, average="micro")
    hamming_loss = metrics.hamming_loss(target, predict)

    return micro_f1, micro_p, micro_r, hamming_loss, one_error


def main():
    parser = argparse.ArgumentParser(description="-----[CNN-classifier]-----")
    parser.add_argument("--mode", default="train", help="train: train (with test) a model / test: test saved models")
    parser.add_argument("--model", default="rand", help="available models: rand, static, non-static, multichannel")
    parser.add_argument("--dataset", default="TREC", help="available datasets: MR, TREC")
    parser.add_argument("--epoch", default=100, type=int, help="number of max epoch")
    parser.add_argument("--learning_rate", default=1e-3, type=float, help="learning rate")
    parser.add_argument("--gpu", default=1, type=int, help="the number of gpu to be used")
    # parser.add_argument("--hidden dim", default=256, type=int, help="the dimension of encoder and decoder")
    options = parser.parse_args()
    data = readdata()

    params = {
        "MODEL": options.model,
        "DATASET": options.dataset,
        "EPOCH": options.epoch,
        "LEARNING_RATE": 0.1,
        "MAX_SENT_LEN": max([len(sent) for sent in data["train_x"] + data["dev_x"] + data["test_x"]]),
        "BATCH_SIZE": 32,
        "WORD_DIM": 128,
        "VOCAB_SIZE": len(data["vocab"]),
        "CLASS_SIZE": len(data["classes"]),
        "FILTERS": [3, 4, 5],
        "FILTER_NUM": [100, 100, 100],
        "DROPOUT_PROB": 0.5,
        "NORM_LIMIT": 3,
        "HIDDEN_DIM": 300,
        "GPU": options.gpu
    }

    print("=" * 20 + "INFORMATION" + "=" * 20)
    print("MODEL:", params["MODEL"])
    print("DATASET:", params["DATASET"])
    print("VOCAB_SIZE:", params["VOCAB_SIZE"])
    print("EPOCH:", params["EPOCH"])
    print("LEARNING_RATE:", params["LEARNING_RATE"])
    print("=" * 20 + "INFORMATION" + "=" * 20)

    print("=" * 20 + "TRAINING STARTED" + "=" * 20)
    train(data, params)
    print("=" * 20 + "TRAINING FINISHED" + "=" * 20)


if __name__ == "__main__":
    main()
