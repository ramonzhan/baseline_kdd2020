# coding: utf-8
import os

rho_list = [1]
# alpha_list = [0.1, 0.2, 0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
# alpha_ban_list = [0.1, 0.2, 0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
# lam2_list = [0.001, 0.002, 0.01, 0.02, 0.1, 0.2, 1]
# 第一次调参
alpha_list = [0.1, 0.2, 0.5, 0.8]
alpha_ban_list = [0.1, 0.2, 0.5, 0.8]
lam2_list = [0.001, 0.002, 0.01, 0.02, 0.1, 0.2, 1]

for alpha in alpha_list:
    for alpha_ban in alpha_ban_list:
        for lam2 in lam2_list:
            os.system("python main_GPU.py --rho 1 --alpha {} --alpha_ban {} --lam2 {}"
                      .format(alpha, alpha_ban, lam2))
