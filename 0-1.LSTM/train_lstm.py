# coding: utf-8
import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import torch
import time
import numpy as np
from lstm.models import ContentLSTM
from tqdm import tqdm
from lstm.datacenter import DataCenter
import random
import gc


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
    # data
    parser.add_argument("--traindict", help="train, key-jd idx, value-skill idx list", type=str, default="skills.pkl")
    parser.add_argument("--testdict", help="test, key-jd idx, value-skill idx list", type=str, default="skills.pkl")
    parser.add_argument("--jdcontent", help="key-jd idx, value-list of token idx", type=str, default="jobdesc.pkl")
    parser.add_argument("--scontent", help="key-s idx, value-list of token idx", type=str, default="skill_vocab.pkl")
    parser.add_argument("--vocab", type=str, default="vocab_token.pkl")
    # network parameters
    parser.add_argument("--wv_dim", help="the dimention of the word embedding", type=int, default=128)
    parser.add_argument("--enc_dim", help="the dimention of the encoder", type=int, default=256)
    parser.add_argument("--batch_size", help="the size of a batch", type=int, default=32)
    parser.add_argument("--batch_size_eval", help="the size of a batch when eval", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--epoch_num", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--gpu_id", help="the id of the gpu", type=int, default=0)
    args = parser.parse_args()
    return args


def around(number, deci=3):
    return np.around(number, decimals=deci)


def fetch(content, ids, max_len, pad):
    """
    fetch node content from homo net(jdjd or ss)
    :param content: dict, key--the idx(jd or skill), value--list of token idx
    :param ids: list of idx of in a batch
    :param max_len: max length of jd or skill
    :param pad: the idx of "<pad>"
    :return: [[]](len outer=batch_size, len inner=max_len), [truelen], True
    """
    src_seq_len, code = [], []
    for idx in ids:
        doc_code = content[idx]
        true_length = len(doc_code)
        src_seq_len.append(true_length)
        doc_code.extend(pad for _ in range(max_len - true_length))
        code.append(doc_code)
    return code, src_seq_len


def main_lstm(args):
    # dir
    results_dir = os.path.join(args.root_dir, args.results)
    model_args = "baseline_lstmbr"
    results_dir = os.path.join(results_dir, model_args)

    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    gpu = torch.device('cuda', args.gpu_id)
    datacenter = DataCenter(args)
    datacenter.load_dataset()
    train_jd_nodes = getattr(datacenter, "train_jdidx")
    test_jd_nodes = getattr(datacenter, "test_jdidx")
    traindict = getattr(datacenter, "labels_dict")
    jdcontent = getattr(datacenter, "jdcontent_dict")
    max_jd_len = getattr(datacenter, "max_jd_len")
    pad_idx = getattr(datacenter, "pad_idx")

    start = time.time()
    model = ContentLSTM(datacenter, args, gpu)
    model.cuda(gpu)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for e in range(args.epoch_num):
        avg_loss = []
        model.train()
        jd_batch_list = datacenter.get_batches(train_jd_nodes)
        num_batch = len(jd_batch_list)

        with tqdm(total=num_batch) as t:
            for batch_d in jd_batch_list:
                optimizer.zero_grad()
                # batch給了个列表: jd batch, skill batch(all)
                t.set_description("Epoch [{}]".format(e))
                label_batch = []
                for node in batch_d:
                    label_batch.append(traindict[node])
                batch_size_l = len(label_batch)
                loss = model.loss(fetch(jdcontent, batch_d, max_jd_len, pad_idx), label_batch, batch_size_l)
                loss.backward()
                t.set_postfix(loss=loss.item())
                optimizer.step()
                avg_loss.append(loss.item())
                t.update()

        # 标签分类结果
        micro_f1, micro_p, micro_r, micro_auc, hamming_loss, ranking_loss, coverage, oneerror = \
            model.evaluate(test_jd_nodes, traindict)
        results_file = "results_lstm.txt"
        minute = np.around((time.time() - start) / 60, decimals=4)
        ls = np.mean(avg_loss)
        print("E:[{}/{}], F1: [{}], P: [{}], R: [{}], AUC: [{}], \n hloss: [{}], rloss: [{}] cov: [{}], oerror: [{}] \t"
              "time: [{} mins]"
              .format(e, around(ls, 2),
                      around(micro_f1), around(micro_p), around(micro_r), around(micro_auc),
                      around(hamming_loss), around(ranking_loss), around(coverage), around(oneerror),
                      minute))

        avg_loss.clear()
        gc.collect()

if __name__ == "__main__":
    main_lstm(parse_args())