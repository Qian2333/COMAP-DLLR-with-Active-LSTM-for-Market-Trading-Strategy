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


def get_rate(l1, l2):
    ans = []
    for i in range(len(l1)):
        ans.append(l2[i] / l1[i])
    return ans


def get_rate2(l1, l2):
    ans = []
    for i in range(len(l1)):
        ans.append(1 - l2[i] / l1[i])
    return ans


def read_txt(path):
    lis1 = []
    lis2 = []
    with open(path, 'r') as f:
        r = f.readlines()
        for s in r:
            s, s1, s2 = s.split(' ')
            if float(s1) + float(s2) > float(s):
                s = float(s1) + float(s2)
            lis1.append(float(s))
            lis2.append(float(s2))
    return get_rate(lis1, lis2)


def read_txt2(path):
    lis1 = []
    lis2 = []
    with open(path, 'r') as f:
        r = f.readlines()
        for s in r:
            s, s1, s2 = s.split(' ')
            if float(s1) + float(s2) > float(s):
                s = float(s1) + float(s2)
            lis1.append(float(s))
            lis2.append(float(s1))
    return get_rate2(lis1, lis2)


if __name__ == '__main__':
    fig = plt.figure(dpi=100, figsize=(15, 5), facecolor='white')
    ax = fig.add_subplot(111)
    line2 = ax.plot(read_txt('tot_money_without_stop1.txt'), label='total assets-without stop loss')
    line2[0].set_linewidth(2)
    line2[0].set_color('orange')
    line2 = ax.plot(read_txt2('tot_money_without_stop1.txt'), label='total assets-without stop loss', linestyle=':')
    line2[0].set_linewidth(2)
    line2[0].set_color('orange')
    # line4 = ax.plot(read_txt('tot_money_without_contain1.txt'), label='total assets-without RPM')
    # line4[0].set_linewidth(1)
    # line4[0].set_color('green')
    # line4 = ax.plot(read_txt2('tot_money_without_contain1.txt'), label='total assets-without RPM', linestyle=':')
    # line4[0].set_linewidth(1)
    # line4[0].set_color('green')
    # line3 = ax.plot(read_txt('tot_money_nosp1.txt'), label='total assets-without RI')
    # line3[0].set_linewidth(1)
    # line3[0].set_color('blue')
    # line3 = ax.plot(read_txt2('tot_money_nosp1.txt'), label='total assets-without RI', linestyle=':')
    # line3[0].set_linewidth(1)
    # line3[0].set_color('blue')
    line1 = ax.plot(read_txt('tot_money1.txt'), label='total assets')
    line1[0].set_linewidth(3)
    line1[0].set_color('red')
    line1 = ax.plot(read_txt2('tot_money1.txt'), label='total assets', linestyle=':')
    line1[0].set_linewidth(3)
    line1[0].set_color('red')
    plt.xlabel('days from 11/11/16', fontsize=15)
    plt.ylabel('dollars', fontsize=15)
    plt.title('total asset', fontsize=20)
    plt.legend()
    plt.show()



