#@Time      :2019/3/31 16:13
#@Author    :zhounan
# @FileName: camel_GPU.py

import numpy as np
import cupy as cp
import time
from camel_utils.util import path_exists
from camel_utils.data import Data
from tqdm import tqdm
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics.pairwise import euclidean_distances
from camel_utils.evaluate import evaluate
import time


def ADMM(y_not_j, y_j, rho, alpha):
    """
    ADMM algriothm for label correlation
    :param y_not_j: cupy, dim {n_instances, n_labels - 1}
        train label data which not contain j col
    :param y_j: cupy, dim {n_instances, 1}
        train label data which only contain j col
    :param rho: ADMM augmented lagrangian parameter(not mentioned in the paper)
    :return: cupy, dim {n_labels - 1, 1}
    """
    #w, _ = cp.linalg.eig(y_not_j.T.dot(y_not_j))
    #rho = 1 / (cp.amax(cp.absolute(w)))
    #rho = 0.05
    max_iter = 1000
    AB1 = 1e-4
    AB2 = 1e-2
    n = y_not_j.shape[1]
    
    S_j = cp.zeros(y_not_j.shape[1]).reshape(-1, 1)
    z = cp.zeros_like(S_j)
    l1norm = cp.max(cp.abs(cp.dot(y_j.T, y_not_j))) / 100
    u = cp.zeros_like(S_j)
    I = cp.eye(y_not_j.shape[1])

    for iter in range(max_iter):
        S_j = cp.dot(cp.linalg.inv(cp.dot(y_not_j.T, y_not_j) + rho * I), cp.dot(y_not_j.T, y_j) + rho*(z - u))

        z_old = z
        omega = l1norm / rho
        S_j_hat = alpha * S_j + (1 - alpha) * z_old
        a = S_j_hat + u
        z = cp.maximum(0, a - omega) - cp.maximum(0, -a - omega)
        u = u + S_j_hat - z

        r_norm = cp.linalg.norm(S_j - z)
        s_norm = cp.linalg.norm(-rho * (z - z_old))
        eps_pri = cp.sqrt(n) * AB1 + AB2 * cp.maximum(cp.linalg.norm(S_j), cp.linalg.norm(-z))
        eps_dual = cp.sqrt(n) * AB1 + AB2 * cp.linalg.norm(rho * u)

        if r_norm < eps_pri and s_norm < eps_dual:
            break
    return z

def CAMEL(S, x, y, alpha, lam2):
    """
    get caml parameter
    :param S: cupy, dim {n_labels, n_labels}
        label correlation matrix
    :param x: numpy, dim {n_instances, n_features}
    :param y: numpy, dim {n_instances, n_labels}
    :param alpha: number
        alpha is the tradeoff parameter that controls the collaboration degree
    :param lam2: number
        λ1 and λ2 are the tradeoff parameters determining the relative importance of the above three terms
        λ1 is given in the paper, equal to 1
    :return:
    """
    max_iter = 51
    num_instances = x.shape[0]
    num_labels = y.shape[1]
    y = cp.array(y)

    G = (1-alpha)*cp.eye(num_labels)+ alpha*S #size num_labels * num_labels
    Z = y #size num_instances * num_labels
    lam1 = 1

    sigma = cp.sum(euclidean_distances(x)) / (num_instances * (num_instances - 1))
    gamma = np.array((1 / (2 * sigma * sigma)).tolist())
    K = cp.array(rbf_kernel(x, gamma=gamma)) #size num_instances * num_instances

    H = (1 / lam2) * K + cp.eye(num_instances) #size num_instances * num_instances

    for iter in range(max_iter):
        #b_T is row vector
        H_inv = cp.linalg.inv(H) #size num_instances * num_instances
        b_T = cp.sum(cp.dot(H_inv, Z), axis=0) / cp.sum(H_inv)
        b_T = b_T.reshape(1, -1)#size 1 * num_labels

        A = cp.dot(H_inv, Z - b_T.repeat(num_instances, axis=0)) #size num_instances * num_labels

        T = (1 / lam2) * cp.dot(K, A) + b_T.repeat(num_instances, axis=0) #size num_instances * num_labels
        Z = cp.dot((T + lam1 * cp.dot(y, G.T)), cp.linalg.inv(cp.eye(num_labels) + lam1 * cp.dot(G, G.T)))

    return G, A, gamma, b_T

def predict(G, A, gamma, x, x_test, b_T, lam2):
    num_instances = x.shape[0]
    num_labels = G.shape[0]
    b = b_T.T
    #a col vector
    K_test = cp.array(rbf_kernel(x, x_test, gamma=gamma)) # size n_train * n_test
    temp = cp.dot(K_test.T, A) #(n_test * n_train) * (n_train * n_label) = (n_test * n_label)
    temp = (1 / lam2) * temp.T + b # (n_label * n_test)

    output = cp.dot(temp.T, G) # (n_label * n_test)
    pred = cp.copy(output)
    pred[pred > 0] = 1
    pred[pred < 0] = -1
    return output, pred

#Mean square error
def loss(predcit, y):
    y = cp.array(y)
    loss = cp.sum(cp.square(y-predcit), axis=1)
    loss = cp.mean(loss)
    return loss.tolist()


def train(x_train, y_train, rho, alpha, alpha_ban, lam2, x_test, y_test):

    num_labels = y_train.shape[1]
    S = cp.zeros((num_labels, num_labels))
    f = open("/home/rleating/kdd2020/results/baseline_camel/rho[{}]_alpha[{}]_alphaban[{}]_lam[{}].txt"
             .format(rho, alpha, alpha_ban, lam2), "w", encoding="utf-8")

    start_time = time.time()
    # get label correlation matrix
    for j in range(num_labels):
        y_j = cp.array(y_train[:, j].reshape(-1, 1))
        y_not_j = cp.array(np.delete(y_train, j, axis=1))
        S_j = ADMM(y_not_j, y_j, rho, alpha_ban)
        S[j, :] = cp.array(np.insert(np.array(S_j.tolist()), j, 0))
        if j % 10 == 0:
            end_time = time.time()
            minute = np.around((end_time - start_time) / 60, decimals=3)
            print("we trained label [{}]th, cost time [{}]min accumtely".format(j, minute))


    # get caml parameter
    G, A, gamma, b_T = CAMEL(S, x_train, y_train, alpha, lam2)

    # evalue
    test_output, test_predict = predict(G, A, gamma, x_train, x_test, b_T, lam2)
    y_test[y_test==-1] = 0
    test_predict[test_predict==-1] = 0
    metric = evaluate(y_test, np.array(test_output.tolist()), np.array(test_predict.tolist()))
    print(metric, file=f)
    print(metric)
    f.close()
