# baseline
> 刘俪婷 张文政  
2020年1月3日

这里用来写论文涉及到的基线算法。包括最基本的基准：LSTM、传统的机器学习多标签算法、端到端的深度学习多标签算法。每个对比算法占一个文件夹，
文件夹命名为x-x.对比算法名。

## 1-1. BR

实现了两个版本，一个使用SVM做基本分类器一个使用NN做基本分类器（仿照skmultilearn写，但由于skmultilearn无法使用GPU训练，故用pytorch重写）。综合考虑时间影响，最终使用NN：单层神经网络+tanh激活函数，epoch=50，lr=1e-3

