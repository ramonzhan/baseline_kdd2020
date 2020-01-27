# coding: utf-8
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from lstm.util import multi_hot
import numpy as np
from sklearn import metrics
from metrics.metrics import Coverage, OneError


def fetch(content, ids, max_len, pad):
    """
    fetch node content from homo net(jdjd or ss)
    :param content: dict, key--the idx(jd or skill), value--list of token idx
    :param ids: list of idx of in a batch
    :param max_len: max length of jd or skill
    :param pad: the idx of "<pad>"
    :return: [[]](len outer=batch_size, len inner=max_len), [truelen], True
    """
    src_seq_len = []
    code = []
    for idx in ids:
        doc_code = content[idx]
        true_length = len(doc_code)
        src_seq_len.append(true_length)
        doc_code.extend(pad for _ in range(max_len - true_length))
        code.append(doc_code)
    return code, src_seq_len


class ContentLSTM(nn.Module):
    def __init__(self, datacenter, config, device, layer=2, bidirect=True):
        super(ContentLSTM, self).__init__()
        self.device = device
        self.config = config
        dropout = 0 if layer == 1 else config.dropout
        self.batch_size = config.batch_size
        self.skill_num = getattr(datacenter, "skill_num")
        self.jdcontent = getattr(datacenter, "jdcontent_dict")
        self.max_jd_len = getattr(datacenter, "max_jd_len")
        self.pad_idx = getattr(datacenter, "pad_idx")
        enc_dim = config.enc_dim // 2 if bidirect else  config.enc_dim
        word_num = getattr(datacenter, "word_num")
        pretrained_w2v = torch.Tensor(getattr(datacenter, "w2v"))
        self.word_embeddings = nn.Embedding(word_num, config.wv_dim).to(device)
        self.word_embeddings.from_pretrained(pretrained_w2v, freeze=False)
        init.xavier_uniform_(self.word_embeddings.weight)

        self.encoder = nn.LSTM(config.wv_dim, enc_dim, num_layers=layer, batch_first=True,
                                   dropout=dropout, bidirectional=bidirect).to(device)
        self.loss_fn = nn.BCEWithLogitsLoss(reduction="sum")
        self.full_layer = nn.Linear(enc_dim * 2, self.skill_num).to(device)
        self.activate_sigmoid = nn.Sigmoid()

    def forward(self, fetch_tuple):
        """
        :param node_batch: [batch_size, word_num]，每一行是一个文本的id list
        :param src_seq_len: 原始句子（未补齐）的长度，list
        :return: 编码后的隐向量
        """
        # 编码从fetch中拿取的东西，第3个和第四个元素分别指示包含的jd个数和s个数
        node_batch = fetch_tuple[0]
        src_seq_len = fetch_tuple[1]
        batch_row_idx = list(range(len(src_seq_len)))
        src_seq_len_1 = [i - 1 for i in src_seq_len]
        input_idx = torch.LongTensor(node_batch).to(self.device)   # 最大索引小于30000，用16位即可。
        embed = self.word_embeddings(input_idx)
        packed_words = pack_padded_sequence(embed, src_seq_len, batch_first=True, enforce_sorted=False)
        lstm_out, hidden = self.encoder(packed_words, None)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        # (batch_size, words_num, num_directions * hidden_size)
        output = lstm_out[batch_row_idx, src_seq_len_1, :]
        output = self.full_layer(output)
        return output

    def loss(self, nodes, labels, bs_l):
        logits = self.forward(nodes)
        labels_target = multi_hot(labels, self.skill_num, bs_l, self.device)
        loss_bce = self.loss_fn(logits, labels_target)
        return loss_bce

    def evaluate(self, test_jd_nodes, labels):
        self.eval()
        # jd_content = pkl.load()
        print("evaluating ... ")
        data_num = len(test_jd_nodes)
        data = test_jd_nodes
        bs = self.config.batch_size_eval
        total_batch = data_num // bs
        predict, target = np.array([0 for _ in range(self.skill_num)]), np.array([0 for _ in range(self.skill_num)])
        confidence = np.array([0 for _ in range(self.skill_num)])
        # batch

        # fetch(jdcontent, batch_d, max_jd_len, pad_idx)

        for batch_idx in range(total_batch):
            start = batch_idx * bs
            end = (batch_idx + 1) * bs
            intance = data[start: end]
            if len(intance) == 0:
                continue
            probs = self.activate_sigmoid(self.forward(fetch(self.jdcontent, intance, self.max_jd_len, self.pad_idx)))

            label = []
            for id in intance:
                label.append(labels[id])
            labels_t = multi_hot(label, self.skill_num, bs, self.device).detach().cpu().numpy()
            probs = probs.detach().cpu().numpy()
            confidence = np.vstack((confidence, probs))
            pred = np.array(list(map(lambda x: list(map(lambda y: 1.0 if y > 0.5 else 0.0, x)), probs)))
            predict = np.vstack((predict, pred))
            target = np.vstack((target, labels_t))

        confidence = np.delete(confidence, 0, 0)
        predict = np.delete(predict, 0, 0)
        target = np.delete(target, 0, 0)
        micro_f1 = metrics.f1_score(target, predict, average="micro")  #
        micro_p = metrics.precision_score(target, predict, average="micro")
        micro_r = metrics.recall_score(target, predict, average="micro")
        micro_auc = metrics.roc_auc_score(target, confidence, average="micro")

        hamming_loss = metrics.hamming_loss(target, predict)
        ranking_loss = metrics.label_ranking_loss(target, confidence)
        coverage = Coverage(confidence, target)
        oneerror = OneError(confidence, target)

        return micro_f1, micro_p, micro_r, micro_auc, hamming_loss, ranking_loss, coverage, oneerror
