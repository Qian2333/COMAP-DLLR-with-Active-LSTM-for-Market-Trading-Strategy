import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from util.toolbag import average_window, linear_regression, linear_regression_point

SEQ_LEN = 28        # 观察期
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
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    # ax.plot(ori_seq, label='real')
    tot = len(seq)
    for i in range(0, tot - step + 1, K_LEN):
        # print(seq)
        k, b = linear_regression(seq[i:(i + step)])
        lin_seq = linear_regression_point(k, b, step, step=(1./3.)).reshape(-1).tolist()
        # print(lin_seq)
        # print(lin_seq)
        # f.write('plot([')
        # for ss in idl[i:(i + step)]:
        #     f.write(str(ss) + ' ')
        # f.write('], [')
        # for ss in lin_seq:
        #     f.write(str(ss) + ' ')
        # f.write('])\n')
        padding = [None for i in range(i * WINDOW_STRIVE)]
        # padding = padding + lin_seq
        rel = []
        # for item in padding:
        #     for _ in range(WINDOW_STRIVE):
        #         rel.append(item)
        # print(padding)
        # print(seq[i:i+step])
        # print(lin_seq)
        ax.plot(padding + lin_seq)
    # f.close()
    plt.legend()
    plt.show()


def get_rate2(ori, step):
    ans = []
    for i in range(len(ori)):
        ans.append(ori[i] / max(ori[max(0, i - step):(i + 1)]) - 1)
    return ans


def get_rate(ori, step):
    ans = []
    for i in range(len(ori)):
        ans.append(ori[i] / ori[max(0, i - step)] - 1)
    return ans


def get_rate3(ori, step):
    ans = [None for i in range(90)]
    import util.toolbag as tbg
    for i in range(90, len(ori)):
        las = max(0, i - step)
        fg = tbg.fg(ori[las:(i + 1)], 3)
        las = ori[las]
        ans.append(((ori[i] - las) / las - 1) / fg)
    return ans


def read_txt(path):
    lis = [1000 for i in range(60)]
    lis = []
    with open(path, 'r') as f:
        r = f.readlines()
        for s in r:
            lis.append(float(s))
    # return lis,
    return get_rate3(lis, 365)


def read_txt1(path):
    lis = [1000 for i in range(60)]
    lis = []
    with open(path, 'r') as f:
        r = f.readlines()
        for s in r:
            s, _, _ = s.split()
            lis.append(float(s))
    # return lis,
    return get_rate3(lis, 365)


if __name__ == '__main__':
    # gold_ori = pd.read_csv('test.csv').get('value').values
    # gold = average_window(gold_ori, WINDOW_LEN, WINDOW_STRIVE)
    # get_k_lis(gold, K_STRIVE, gold_ori)
    # plot_results(lis)
    fig = plt.figure(dpi=100, figsize=(15, 5), facecolor='white')
    ax = fig.add_subplot(111)
    line1 = ax.plot(read_txt1('tot_money_99.txt'), label='Transaction costs 1% for bitcoin and 0.5% for gold')
    line2 = ax.plot(read_txt('tot_money.txt'), label='Transaction costs 2% for bitcoin and 1% for gold')
    line4 = ax.plot(read_txt1('tot_money_96.txt'), label='Transaction costs 4% for bitcoin and 2% for gold')
    line3 = ax.plot(read_txt1('tot_money_94.txt'), label='Transaction costs 6% for bitcoin and 3% for gold')
    line1[0].set_linewidth(1)
    line1[0].set_color('red')
    line2[0].set_linewidth(1)
    line2[0].set_color('orange')
    line3[0].set_linewidth(1)
    line3[0].set_color('blue')
    line4[0].set_linewidth(1)
    line4[0].set_color('green')
    plt.xlabel('Days from 9/11/16 to 9/10/21', fontsize=15)
    plt.ylabel('Annualized RI', fontsize=15)
    # plt.title('total asset', fontsize=20)
    plt.legend()
    plt.show()


