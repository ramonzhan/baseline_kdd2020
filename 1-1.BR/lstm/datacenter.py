# coding : utf-8
import os
import pickle as pkl
import numpy as np


class DataCenter(object):
    def __init__(self, config):
        super(DataCenter, self).__init__()
        self.config = config
        self.root_dir = os.path.join(config.root_dir, config.dataset, config.data_num)

    def load_dataset(self):
        jd_content_file = os.path.join(self.root_dir, self.config.jdcontent)
        traindict_dir = os.path.join(self.root_dir, self.config.traindict)
        s_content_file = os.path.join(self.root_dir, self.config.scontent)
        token_vocab_dir = os.path.join(self.root_dir, self.config.vocab)
        jdcontent, max_jd_len, pad_idx = pkl.load(open(jd_content_file, "rb"))
        train_dict = pkl.load(open(traindict_dir, "rb"))  # dict, key: jd idx, value: corr skill idx list
        token_list = pkl.load(open(token_vocab_dir, "rb"))
        scontent, max_s_len, skill_num = pkl.load(open(s_content_file, "rb"))
        word_num = len(token_list)
        test_indexs, train_indexs = self.split_data()
        setattr(self, 'test_jdidx', test_indexs)
        setattr(self, 'train_jdidx', train_indexs)
        setattr(self, 'jdcontent_dict', jdcontent)
        setattr(self, 'labels_dict', train_dict)
        setattr(self, 'word_num', word_num)
        setattr(self, 'max_jd_len', max_jd_len)
        setattr(self, 'pad_idx', pad_idx)
        setattr(self, "skill_num", skill_num)

    def split_data(self):
        dataset_dir = os.path.join(self.root_dir, "train_test.pkl")
        train_indexs, test_indexs = pkl.load(open(dataset_dir, "rb"))
        print("train set contains [{}] nodes, test set contains [{}] nodes"
              .format(len(train_indexs), len(test_indexs)))
        print("<---------------------------------------------------------->")
        return test_indexs, train_indexs

    def get_batches(self, node_indexs):
        np.random.shuffle(node_indexs)
        node_num = len(node_indexs)
        num_batches = node_num // self.config.batch_size
        batch_list = []
        for n in range(num_batches):
            batch_list.append(node_indexs[n * self.config.batch_size: (n + 1) * self.config.batch_size])
        return batch_list
