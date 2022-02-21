import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from util.toolbag import average_window, linear_regression, linear_regression_point

SEQ_LEN = 60        # 观察期
THRESHOLD1 = 3      # 牛熊阈值
THRESHOLD2 = 0.2    # 可行性阈值
BETA = 0.2          # 买入比例
ALPHA1 = 0.02       # 比特币买入佣金
ALPHA2 = 0.01       # 黄金买入佣金
WINDOW_LEN = 3      # 窗口大小
WINDOW_STRIVE = 3   # 窗口步长
K_LEN = 1        # 斜率拟合长度
K_STRIVE = 3        # 斜率拟合长度


def plot_results(data):
    fig = plt.figure(dpi=50, figsize=(21, 7), facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(data, label='total assets')
    plt.legend()
    plt.show()


def get_k_lis(seq, step, ori_seq):
    # f = open('plot.txt', 'w')
    # idl = [i for i in range(len(seq))]
    # f.write('plot([')
    # for s in idl:
    #     f.write(str(s) + ' ')
    # f.write('], [')
    # for s in seq:
    #     f.write(str(s) + ' ')
    # f.write('])\n')
    # f.write('hold on\n')
    fig = plt.figure(dpi=100, figsize=(15, 5), facecolor='white')
    # ax = fig.add_subplot(111)
    # print(ori_seq.tolist())
    plt.plot([None for i in range(400)] + ori_seq.tolist(), color='blue',
             linewidth=1, linestyle='--', label='real data')
    # ax.plot(ori_seq, label='real')
    tot = len(seq)
    for i in range(0, tot - step + 1, K_LEN):
        # print(seq)
        k, b = linear_regression(seq[i:(i + step)])
        lin_seq = linear_regression_point(k, b, step, step=(1 / 3)).reshape(-1).tolist()
        # print(lin_seq)
        # print(lin_seq)
        # f.write('plot([')
        # for ss in idl[i:(i + step)]:
        #     f.write(str(ss) + ' ')
        # f.write('], [')
        # for ss in lin_seq:
        #     f.write(str(ss) + ' ')
        # f.write('])\n')
        padding = [None for i in range(i * WINDOW_STRIVE + 400)]
        # padding = padding + lin_seq
        rel = []
        # for item in padding:
        #     for _ in range(WINDOW_STRIVE):
        #         rel.append(item)
        # print(padding)
        # print(seq[i:i+step])
        # print(lin_seq)
        plt.plot(padding + lin_seq, linewidth=3)
    # f.close()
    plt.xlabel('days from 9/11/16', fontsize=15)
    plt.ylabel('dollars', fontsize=15)
    plt.legend()
    plt.show()


def read_txt(path):
    lis = []
    with open(path, 'r') as f:
        r = f.readlines()
        for s in r:
            lis.append(float(s))
    return lis


if __name__ == '__main__':
    gold_ori = pd.read_csv('test.csv').get('value').values[400:600]
    gold = average_window(gold_ori, WINDOW_LEN, WINDOW_STRIVE)
    get_k_lis(gold, K_STRIVE, gold_ori)


