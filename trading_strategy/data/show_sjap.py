import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from util.toolbag import average_window, linear_regression, linear_regression_point


def plot_results(data):
    fig = plt.figure(dpi=50, figsize=(21, 7), facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(data, label='total assets')
    plt.legend()
    plt.show()


def get_rate(ori, step):
    ans = [None for i in range(60)]
    import util.toolbag as tbg
    for i in range(len(ori)):
        las = max(0, i - step)
        fg = tbg.fg(ori[las:(i + 1)], 3)
        las = ori[las]
        ans.append(((ori[i] - las) / las - 1) / fg)
    return ans


def read_txt(path):
    lis = []
    with open(path, 'r') as f:
        r = f.readlines()
        for s in r:
            lis.append(float(s))
    return get_rate(lis, 365)


if __name__ == '__main__':
    fig = plt.figure(dpi=100, figsize=(15, 5), facecolor='white')
    ax = fig.add_subplot(111)
    line2 = ax.plot(read_txt('tot_money_without_stop.txt'), label='Without stop-loss method')
    line3 = ax.plot(read_txt('tot_money_nosp.txt'), label='Without investment ratio optimization')
    line4 = ax.plot(read_txt('tot_money_without_contain.txt'), label='Without risk parity methodology')
    line1 = ax.plot(read_txt('tot_money.txt'), label='Best model')
    line1[0].set_linewidth(2)
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



