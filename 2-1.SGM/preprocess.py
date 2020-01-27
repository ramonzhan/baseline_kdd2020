# coding: utf-8
import os
import argparse
import utils
import pickle as pkl
import operator


parser = argparse.ArgumentParser(description='preprocess.py')
parser.add_argument('-load_data', type=str, default="/kdd2020/dataset/it_1/")
parser.add_argument('-save_data', type=str, default="/kdd2020/dataset/sgm/it_1/")
parser.add_argument('-src_suf', default='src', help="the suffix of the source filename")
parser.add_argument('-tgt_suf', default='tgt', help="the suffix of the target filename")
opt = parser.parse_args()


def makevocab_src(src_dict, vocab):
    """
    制作输入词典，SGM需求。在我们的任务中就是工作描述的词典
    :param src_dict: dict, key--idx，value---str，以空格分隔的token
    :param vocab: 之前初始化的vocab，里面本身就有BOS和EOS
    :return: 制作好的输入词典
    """
    for sent in src_dict.values():
        tokens = sent.strip().split(" ")
        for word in tokens:
            vocab.add(word)
    return vocab


def makevocab_tgt(tgt_dict, vocab):
    """
    制作输出词典，在我们的任务中就是技能的词典。包含频率
    :param tgt_dict: dict key---jd idx, value ---用空格分割的对应的技能序列
    :param vocab: 之前初始化的vocab，里面本身就有bos和eos
    :return: 制作好的输出词典
    """
    for new_token in tgt_dict.values():
        skill_list = new_token.split(" ")
        for skill in skill_list:
            vocab.add(skill)
    return vocab


def makedata(src_dict, tgt_dict, convert_src, convert_tgt, save_src, save_tgt):
    """
    保存训练以及测试使用的文件，训练则输入训练，测试则输入测试
    :param src_dict. dict key--jd idx 每个元素都是用空格分好的token序列
    :param tgt_dict. dict, key -- jd idx 按顺序的每个jd对应的技能，空格分开
    :param convert_src: 之前保存过的src的字典
    :param convert_tgt: 之前保存过的tgt的字典
    :param save_src: 训练/测试src保存位置
    :param save_tgt: 训练/测试tgt保存位置
    """
    srcid = open(save_src + '.id', 'w')
    tgtid = open(save_tgt + '.id', 'w')
    srcstr = open(save_src + '.str', 'w', encoding='utf8')
    tgtstr = open(save_tgt + '.str', 'w', encoding='utf8')
    jd_idx_list = list(src_dict.keys())   # 所有的jd索引
    sizes = 0
    for jd_idx in jd_idx_list:
        src, tgt = src_dict[jd_idx], tgt_dict[jd_idx]
        sline, tline = src.lower(), tgt.lower()   # str
        src_tokens, tgt_tokens = sline.split(" "), tline.split(" ")
        srcids = convert_src.convertToIdx(src_tokens, utils.UNK_WORD)
        tgtids = convert_tgt.convertToIdx(tgt_tokens, utils.UNK_WORD, utils.BOS_WORD, utils.EOS_WORD)
        # 文件保存位置
        srcid.write(" ".join(list(map(str, srcids)))+'\n'), tgtid.write(" ".join(list(map(str, tgtids)))+'\n')
        srcstr.write(" ".join(src_tokens)+'\n'), tgtstr.write(" ".join(tgt_tokens)+'\n')
        sizes += 1
    srcstr.close(), tgtstr.close(), srcid.close(), tgtid.close()

    return {'srcF': save_src + '.id', 'tgtF': save_tgt + '.id',
            'original_srcF': save_src + '.str', 'original_tgtF': save_tgt + '.str',
            'length': sizes}


def idx2token(input_dict, token_list, sep=" "):
    """
    SGM需要做一个输入的词典和输出的词典，我门先将原先的数据恢复成具体的token
    :param input_dict: key--idx，value--list,元素为token的id
    :param token_list: 之前制作好的词典，列表
    :param sep: 恢复之后以sep间隔两个token
    :return: dict, key--idx, value--str, 以空格间隔的具体token
    """
    token_vocab = dict(zip(range(len(token_list)), token_list))
    input_dict_copy = input_dict.copy()
    output_dict = dict()
    for idx, token_idx_list in input_dict_copy.items():
        new_token_list = []
        for token_idx in token_idx_list:
            new_token_list.append(token_vocab[token_idx])
        output_dict[idx] = sep.join(new_token_list)
    return output_dict


def idx2skill(ins_dict):
    """
    SGM需要统计每个输出标签的频率以便排序，我们先将原先的jd-s字典转成jd和具体的skill  // 我们直接用索引来代替其实是一样的
    :param ins_dict: key--jd idx， value--list，对应的skill idx
    :param skill_vocab: dict key---skill idx, value--具体的技能
    :return: dict，key--jd idx， value--空格分隔的skill string
    """
    output_dict = dict()
    for jd_idx, skills_idxlist in ins_dict.items():
        skills_idxlist = list(map(str, skills_idxlist))
        skills_list = []
        # for skill_idx in skills_idxlist:
        #     skills_list.append(skill_vocab[skill_idx])
        output_dict[jd_idx] = " ".join(skills_idxlist)
    return output_dict


def traintest_split(src_dict, tgt_dict, train_idxs, test_idxs):
    """
    根据已有的train/test索引分别制作训练和测试用的dict（tgt和src）
    :param src_dict: 总的src dict key-jdidx，value对应的内容，
    :param tgt_dict: 总的tgt，key--jdidx，value对应的标签
    :param train_idxs: list，用于训练的jdidx
    :param test_idxs: list 用于测试的idx
    :return: train_src_dict, test_src_dict, train_tgt_dict, test_tgt_dict
    """
    train_src_dict, test_src_dict, train_tgt_dict, test_tgt_dict = dict(), dict(), dict(), dict()
    for train_idx in train_idxs:
        train_src_dict[train_idx] = src_dict[train_idx]
        train_tgt_dict[train_idx] = tgt_dict[train_idx]
    for test_idx in test_idxs:
        test_src_dict[test_idx] = src_dict[test_idx]
        test_tgt_dict[test_idx] = tgt_dict[test_idx]
    return train_src_dict, test_src_dict, train_tgt_dict, test_tgt_dict


def main():
    if not os.path.exists(opt.save_data):
        os.mkdir(opt.save_data)
    # dir
    dataset_dir = os.path.join(opt.load_data, "train_test.pkl")
    jobdesc_dir = os.path.join(opt.load_data, "jobdesc.pkl")
    skills_dir = os.path.join(opt.load_data, "skills.pkl")
    skill_vocab_dir = os.path.join(opt.load_data, 'skill_vocab.pkl')
    token_vocab_dir = os.path.join(opt.load_data, "vocab_token.pkl")
    save_train_src_dir = os.path.join(opt.save_data, "train" + opt.src_suf)
    save_train_tgt_dir = os.path.join(opt.save_data, "train" + opt.tgt_suf)
    save_test_src_dir = os.path.join(opt.save_data, "test" + opt.src_suf)
    save_test_tgt_dir = os.path.join(opt.save_data, "test" + opt.tgt_suf)
    src_dict_dir = os.path.join(opt.save_data, "src.dict")
    tgt_dict_dir = os.path.join(opt.save_data, "tgt.dict")
    label_json_dir = os.path.join(opt.save_data, "label_sorted.json")

    # load data
    train_indexs, test_indexs = pkl.load(open(dataset_dir, "rb"))
    token_vocab = pkl.load(open(token_vocab_dir, "rb"))
    skill_vocab, max_s_len, skill_num = pkl.load(open(skill_vocab_dir, "rb"))
    jdcontent, max_jd_len, pad_idx = pkl.load(open(jobdesc_dir, "rb"))
    skill_dict = pkl.load(open(skills_dir, "rb"))    # key: jd idx, value: 对应的技能idx列表

    print("convert token idx to real token")
    jdcontent, skill_vocab = idx2token(jdcontent, token_vocab), idx2token(skill_vocab, token_vocab, sep="")
    tgt_dict = idx2skill(skill_dict)

    print("split the train and test set...")
    train_src, test_src, train_tgt, test_tgt = traintest_split(jdcontent, tgt_dict, train_indexs, test_indexs)

    print('Building source vocabulary...')
    dicts = dict()
    dicts['src'] = utils.Dict([utils.PAD_WORD, utils.UNK_WORD, utils.BOS_WORD, utils.EOS_WORD])
    dicts['src'] = makevocab_src(jdcontent, dicts['src'])
    print('Building target vocabulary...')
    dicts['tgt'] = utils.Dict([utils.PAD_WORD, utils.UNK_WORD, utils.BOS_WORD, utils.EOS_WORD])
    dicts['tgt'] = makevocab_tgt(tgt_dict, dicts['tgt'])

    print('Preparing training ...')
    train = makedata(train_src, train_tgt, dicts['src'], dicts['tgt'], save_train_src_dir, save_train_tgt_dir)
    print('Preparing testing ...')
    test = makedata(test_src, test_tgt, dicts['src'], dicts['tgt'], save_test_src_dir, save_test_tgt_dir)

    print('Saving source vocabulary to \'' + src_dict_dir + '\'...')
    dicts['src'].writeFile(src_dict_dir)
    print('Saving source vocabulary to \'' + tgt_dict_dir + '\'...')
    dicts['tgt'].writeFile(tgt_dict_dir)
    data = {'train': train, 'valid': test, 'test': test, 'dict': dicts}
    pkl.dump(data, open(os.path.join(opt.save_data, "data.pkl"), 'wb'))

    # sort the label based on frequency
    freq_dict = dicts['tgt'].frequencies
    skill_freq_dict = dict()
    for skill in skill_vocab.keys():
        skill = str(skill)
        skill_freq_dict[skill] = freq_dict[dicts['tgt'].labelToIdx[skill]]

    sorted_zip = sorted(skill_freq_dict.items(), key=operator.itemgetter(1), reverse=True)

    sorted_skill_dict = dict()
    i = 0
    for skill ,freq in sorted_zip:
        sorted_skill_dict[str(skill)] = i
        i += 1

    f = open(label_json_dir, 'w')
    f.write(str(sorted_skill_dict).replace("'", "\""))
    f.close()


if __name__ == "__main__":
    main()
