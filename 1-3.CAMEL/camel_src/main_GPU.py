#@Time      :2019/3/29 10:44
#@Author    :zhounan
# @FileName: main.py
#import numpy as cp
import cupy as cp
from sklearn.metrics.pairwise import rbf_kernel
from camel_utils.data import Data
from camel_src import camel_GPU
import numpy as np
from decimal import Decimal


def train_val():
    # trade-off para
    rho_list = [1]
    alpha_list = cp.arange(0, 1.1, 0.1)
    alpha_ban_list = cp.arange(0, 1.1, 0.1)
    lam2_list = cp.array([0.001, 0.002, 0.01, 0.02, 0.1, 0.2, 1])

    datasets = ['yeast', 'scene', 'enron', 'image']
    dataset = datasets[2]
    data = Data(dataset, label_type=0)
    x, y = data.load_data()
    camel_GPU.train_val(dataset, x, y, rho_list, alpha_list, alpha_ban_list, lam2_list)

def train():
    dataset = "doc2vec/it_1"
    data = Data(dataset, label_type=0, path="/home/rleating/kdd2020/dataset/")
    x, y, x_test, y_test = data.load_data_separated()
    camel_GPU.train(x, y, rho=1, alpha=0.2, alpha_ban=0.5, lam2=0.001, x_test=x_test, y_test=y_test)


if __name__ == '__main__':
    train()
