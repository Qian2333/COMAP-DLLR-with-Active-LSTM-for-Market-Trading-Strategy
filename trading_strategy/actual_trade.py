import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from util.toolbag import average_window, linear_regression, \
    linear_regression_point
import util.toolbag as tbg
import math

SEQ_LEN = 60        # 观察期
THRESHOLD1 = 3      # 牛熊阈值
THRESHOLD2 = 0.2    # 可行性阈值
BETA = 0.9          # 买入比例
ALPHA1 = 0.98       # 比特币买入佣金
ALPHA11 = 0.98 * 0.98       # 比特币买入佣金
ALPHA2 = 0.99       # 黄金买入佣金
ALPHA22 = 0.99 * 0.99       # 比特币买入佣金
WINDOW_LEN = 3      # 窗口大小
WINDOW_STRIVE = 3   # 窗口步长
K_LEN = 1        # 斜率拟合长度
K_STRIVE = 3        # 斜率拟合长度
CONTAIN = 0.9        # 斜率拟合长度


"""
比特币 seq=30, window = 3      53000
"""


def plot_results(data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot()
    ax.plot(data)
    plt.legend()
    plt.show()


def buy_or_not1(las30, threshold):
    # today = las30[-1]
    # las30 = average_window(las30, WINDOW_LEN, WINDOW_STRIVE)
    # las301 = las30 / las30[0] - 1
    # # print(las30)
    # # input()
    # k, b = linear_regression(las301)
    # if k > 0.03:
    #     # print('fuck')
    #     return -1
    # elif k < -0.03:
    #     # print('fuck')
    #     return 1
    # mse = tbg.cal_mse(las30)
    # mid = sum(las30) / len(las30)
    # if today > mid + mse:
    #     return -1
    # elif today < mid - mse:
    #     return 1
    # else:
    #     return 0

    # k, b = linear_regression(las30)
    # len1 = len(las30)
    # las30p = linear_regression_point(k, b, length=len1).reshape(-1)
    # las30 = np.array(las30)
    # mse = (las30p - las30)
    # mse = np.dot(mse, mse)
    # mse /= len1
    # mse = math.sqrt(mse)
    # if las30p[-1] + mse < las30[-1]:
    #     return -1
    # elif las30p[-1] - mse > las30[-1]:
    #     return 1
    # else:
    #     return 0
    # print(las30)

    las30 = las30 / las30[0] - 1
    k_lis = []
    for i in range(len(las30) - K_STRIVE + 1):
        k, b = linear_regression(las30[i:(i + K_STRIVE)])
        k_lis.append(k)
    now = k_lis[-1]
    k_tot, _ = linear_regression(las30)
    if k_tot > threshold:
        return 2
    elif k_tot < -threshold:
        return 2
    # if now > threshold:
    #     return 2
    # elif now < -threshold:
    #     return 2
    k_lis = k_lis[:-1]
    average_k = sum(k_lis) / len(k_lis)
    mse_k = tbg.cal_mse(k_lis)
    if now < average_k + mse_k:
        return 1
    elif now > average_k - mse_k:
        return -1
    else: return 0


# def buy_or_not1(las30):
#     las30 = average_window(las30, WINDOW_LEN, WINDOW_STRIVE)
#     # print(las30)
#     las30 = las30 / las30[0] - 1
#     # print(las30)
#     # input()
#     k_lis = []
#     for i in range(len(las30) - K_STRIVE + 1):
#         k, b = linear_regression(las30[i:(i + K_STRIVE)])
#         k_lis.append(k)
#     now = k_lis[-1]
#     if now > 0.20:
#         print('fuck')
#         return -1
#     elif now < -0.20:
#         print('fuck')
#         return 1
#     k_lis = k_lis[:-1]
#     average_k = sum(k_lis) / len(k_lis)
#     mse_k = tbg.cal_mse(k_lis)
#     if now < average_k + mse_k:
#         return 1
#     elif now > average_k - mse_k:
#         return -1
#     else: return 0


def sigmoid(x):
    return 1 / (1 + np.exp(x))


def buy_or_not2(idd, sett, opt, threshold):
    if idd > 1815 and opt == 1:
        return False
    if idd > 1815 and opt == -1:
        return True
    if opt == -1:
        if sigmoid(sett[idd] * threshold) > 0.8:
            return True
        else:
            return False
    if opt == 1:
        if sigmoid(sett[idd] * threshold) < 0.2:
            return True
        else:
            return False


if __name__ == '__main__':
    gold = pd.read_csv('data/gold.csv').get('value').values
    gold_op = pd.read_csv('data/gold.csv').get('op').values
    bitb = pd.read_csv('data/test.csv').get('value').values
    lis_own_b = []
    lis_own_g = []
    own_tot = [0, 0]
    tot_money = 1000
    res = 1000
    pridect_k = tbg.read_pre('data/pred_k6.txt')
    pridect_g = tbg.read_pre('data/gold_57.txt')
    ema_k9 = tbg.ema(bitb, 9)
    ema_k12 = tbg.ema(bitb, 12)
    ema_k26 = tbg.ema(bitb, 26)
    ema_g9 = tbg.ema(gold, 9)
    ema_g12 = tbg.ema(gold, 12)
    ema_g26 = tbg.ema(gold, 26)
    macd_k = 224. / 51. * ema_k9 - 16. / 3 * ema_k12 + 16 / 17 * ema_k26
    macd_g = 224. / 51. * ema_g9 - 16. / 3 * ema_g12 + 16 / 17 * ema_g26
    # f_k = open('data/trade_k.txt', 'w')
    # f_g = open('data/trade_g.txt', 'w')
    f_t = open('data/tot_money_c+5.txt', 'w')
    # f_t = open('data/.txt', 'w')
    sold_cg = 0
    sold_cg1 = 0
    sold_cg2 = 0
    sold_ck = 0
    sold_ck1 = 0
    sold_ck2 = 0
    parc_cg = 0
    parc_ck = 0

    for i in range(SEQ_LEN - 1, len(gold)):
        las30 = bitb[(i - SEQ_LEN + 1):(i + 1)]
        opt = buy_or_not1(las30, 0.01)   # 0.01
        if abs(opt) == 2:
            if macd_k[i] > 0 and opt < 0:
                opt = -1
            elif macd_k[i] < 0 and opt > 0:
                opt = 1
        opt_k = False
        opt_g = False
        if opt == 1 and buy_or_not2(i, pridect_k, 1, 100) and \
                own_tot[0] * bitb[i] < CONTAIN * tot_money:# 100
        # if opt == 1 and buy_or_not2(i, pridect_k, 1, 100):
            if i > 1815 or pridect_k[i] * ALPHA1 * ALPHA1 > 0.001:
                opt_k = True
        if opt == -1 and buy_or_not2(i, pridect_k, -1, 100):
            for j in range(len(lis_own_b) - 1, -1, -1):
                bit_bb = lis_own_b[j]
                if bit_bb[1] / bitb[i] > ALPHA1 * ALPHA1:
                    money_get = bit_bb[0] * bitb[i]
                    res += money_get
                    lis_own_b.pop(j)
                    own_tot[0] -= bit_bb[0]
                    sold_ck += 1
                    # f_k.write(str(round(own_tot[0], 2)) + ' ' + str(round(own_tot[0] * bitb[i], 2)) +
                    #           ' | sold: day ' + str(i) + ' contain: ' + str(round(bit_bb[0], 2)) +
                    #           ' in_value: ' + str(round(bit_bb[1], 2)) + ' now: ' + str(bitb[i]) +
                    #           ' earn:' + str(round(bitb[i] * bit_bb[0] - bit_bb[1] * bit_bb[0] / ALPHA11, 2))
                    #           + '\n')
        for j in range(len(lis_own_b) - 1, -1, -1):
            lis_own_b[j][2] = max(lis_own_b[j][2], bitb[i])
            bit_bb = lis_own_b[j]
            if bit_bb[2] / bitb[i] * 0.90 > ALPHA1 * ALPHA1:
                money_get = bit_bb[0] * bitb[i]
                res += money_get
                # if money_get < 0:
                #     print(i, '1')
                #     input()
                lis_own_b.pop(j)
                sold_ck1 += 1
                own_tot[0] -= bit_bb[0]
                # f_k.write(str(round(own_tot[0], 2)) + ' ' + str(round(own_tot[0] * bitb[i], 2)) +
                #           ' | sold: day ' + str(i) + ' contain: ' + str(round(bit_bb[0], 2)) +
                #           ' in_value: ' + str(round(bit_bb[1], 2)) + ' now: ' + str(bitb[i]) +
                #           ' earn:' + str(round(bitb[i] * bit_bb[0] - bit_bb[1] * bit_bb[0] / ALPHA11, 2))
                #           + '\n')
            # print('sold bitb ', money_get, ' ', get_tr)
        if int(gold_op[i]) == 0:
            las30 = gold[(i - SEQ_LEN + 1):(i + 1)]
            opt = buy_or_not1(las30, 0.01)  #0.01
            if abs(opt) == 2:
                if macd_g[i] > 0 and opt < 0:
                    opt = -1
                elif macd_g[i] < 0 and opt > 0:
                    opt = 1
            if opt == 1 and buy_or_not2(i, pridect_g, 1, 200) and \
                    own_tot[1] * gold[i] < CONTAIN * tot_money: # 200
            # if opt == 1 and buy_or_not2(i, pridect_g, 1, 200):
                if i > 1815 or pridect_g[i] * ALPHA2 * ALPHA2 > 0.001:
                    opt_g = True
            if opt == -1 and buy_or_not2(i, pridect_g, -1, 50):
                for j in range(len(lis_own_g) - 1, -1, -1):
                    gold_t = lis_own_g[j]
                    if gold_t[1] / gold[i] > ALPHA2 * ALPHA2:
                        money_get = gold_t[0] * gold[i]
                        # if money_get < 0:
                        #     print(i, '2')
                        #     input()
                        res += money_get
                        sold_cg += 1
                        lis_own_g.pop(j)
                        own_tot[1] -= gold_t[0]
                        # f_k.write(str(round(own_tot[1], 2)) + ' ' + str(round(own_tot[1] * gold[i], 2)) +
                        #           ' | sold: day ' + str(i) + ' contain: ' + str(round(gold_t[0], 2)) +
                        #           ' in_value: ' + str(round(gold_t[1], 2)) + ' now: ' + str(gold[i]) +
                        #           ' earn:' + str(round(gold[i] * gold_t[0] - gold_t[1] * gold_t[0] / ALPHA11, 2))
                        #           + '\n')
            for j in range(len(lis_own_g) - 1, -1, -1):
                lis_own_g[j][2] = max(lis_own_g[j][2], gold[i])
                gold_t = lis_own_g[j]
                if gold_t[2] / gold[i] * 0.93 > ALPHA2 * ALPHA2:
                    money_get = gold_t[0] * gold[i]
                    # if money_get < 0:
                    #     print(i, '2')
                    #     input()
                    res += money_get
                    lis_own_g.pop(j)
                    own_tot[1] -= gold_t[0]
                    sold_cg1 += 1
                    # f_k.write(str(round(own_tot[1], 2)) + ' ' + str(round(own_tot[1] * gold[i], 2)) +
                    #           ' | sold: day ' + str(i) + ' contain: ' + str(round(gold_t[0], 2)) +
                    #           ' in_value: ' + str(round(gold_t[1], 2)) + ' now: ' + str(gold[i]) +
                    #           ' earn:' + str(round(gold[i] * gold_t[0] - gold_t[1] * gold_t[0] / ALPHA11, 2))
                    #           + '\n')
        srg = pridect_g[min(i, 1815)] / tbg.fg(gold[(i - SEQ_LEN + 1):(i + 1)], WINDOW_LEN)
        srk = pridect_k[min(i, 1815)] / tbg.fg(bitb[(i - SEQ_LEN + 1):(i + 1)], WINDOW_LEN)
        # b1 = min((srk / (srg + srk)), 1)
        # b2 = min((srg / (srg + srk)), 1)
        b1 = BETA * srk / (srg + srk)
        b2 = BETA * srg / (srg + srk)
        # b1 = BETA
        # b2 = BETA
        if opt_g and not opt_k:
            b1 = max(b1, 1 - CONTAIN)
            if own_tot[0] * bitb[i] / tot_money > b1:
                for j in range(len(lis_own_b) - 1, -1, -1):
                    lis_own_b[j][2] = max(lis_own_b[j][2], bitb[i])
                    bit_bb = lis_own_b[j]
                    if bit_bb[1] / bitb[i] > ALPHA1 * ALPHA1:
                        money_get = bit_bb[0] * bitb[i]
                        if (own_tot[0] - bit_bb[0]) * bitb[i] / tot_money < b1:
                            sold = own_tot[0] - b1 * tot_money / bitb[i]
                            bit_bb[0] -= sold
                            # if sold * bitb[i] < 0:
                            #     print(i, '1')
                            #     input()
                            res += sold * bitb[i]
                            own_tot[0] -= sold
                            # f_k.write(str(round(own_tot[0], 2)) + ' ' + str(round(own_tot[0] * bitb[i], 2)) +
                            #           ' | sold: day ' + str(i) + ' contain: ' + str(round(bit_bb[0], 2)) +
                            #           ' in_value: ' + str(round(bit_bb[1], 2)) + ' now: ' + str(bitb[i]) +
                            #           ' earn:' + str(round(bitb[i] * bit_bb[0] - bit_bb[1] * bit_bb[0] / ALPHA11, 2))
                            #           + '\n')
                            break
                        # if money_get < 0:
                        #     print(i, '1')
                        #     input()
                        res += money_get
                        lis_own_b.pop(j)
                        own_tot[0] -= bit_bb[0]
                sold_ck2 += 1
                        # f_k.write(str(round(own_tot[0], 2)) + ' ' + str(round(own_tot[0] * bitb[i], 2)) +
                        #           ' | sold: day ' + str(i) + ' contain: ' + str(round(bit_bb[0], 2)) +
                        #           ' in_value: ' + str(round(bit_bb[1], 2)) + ' now: ' + str(bitb[i]) +
                        #           ' earn:' + str(round(bitb[i] * bit_bb[0] - bit_bb[1] * bit_bb[0] / ALPHA11, 2))
                        #           + '\n')
            use_money = max(min(res * b2, CONTAIN * tot_money - own_tot[1] * gold[i]), 0)
            res -= use_money
            get_tr = use_money / gold[i] * ALPHA2 * ALPHA2
            lis_own_g.append([get_tr, gold[i], gold[i]])
            own_tot[1] += get_tr
            parc_cg += 1
            # f_g.write(str(round(own_tot[1], 2)) + ' ' + str(round(own_tot[1] * gold[i], 2)) +
            #           ' | parc: day ' + str(i) +
            #           ' contain: ' + str(round(get_tr, 2)) +
            #           ' value: ' + str(gold[i]) + '\n')
        elif opt_k and not opt_g:
            b2 = max(b2, 1 - CONTAIN)
            if int(gold_op[i]) == 0 and own_tot[1] * gold[i] / tot_money > b2:
                sold_cg2 += 1
                for j in range(len(lis_own_g) - 1, -1, -1):
                    lis_own_g[j][2] = max(lis_own_g[j][2], gold[i])
                    gold_t = lis_own_g[j]
                    if gold_t[1] / gold[i] > ALPHA2 * ALPHA2:
                        money_get = gold_t[0] * gold[i]
                        if (own_tot[1] - gold_t[0]) * gold[i] / tot_money < b2:
                            sold = own_tot[1] - b2 * tot_money / gold[i]
                            gold_t[0] -= sold
                            # if sold * gold[i] < 0:
                            #     print(i, '1')
                            #     input()
                            res += sold * gold[i]
                            own_tot[1] -= sold
                            # f_k.write(str(round(own_tot[1], 2)) + ' ' + str(round(own_tot[1] * gold[i], 2)) +
                            #           ' | sold: day ' + str(i) + ' contain: ' + str(round(gold_t[0], 2)) +
                            #           ' in_value: ' + str(round(gold_t[1], 2)) + ' now: ' + str(gold[i]) +
                            #           ' earn:' + str(round(gold[i] * gold_t[0] - gold_t[1] * gold_t[0] / ALPHA11, 2))
                            #           + '\n')
                            break
                        # if money_get < 0:
                        #     print(i, '1')
                        #     input()
                        res += money_get
                        lis_own_g.pop(j)
                        own_tot[1] -= gold_t[0]
                        # f_k.write(str(round(own_tot[1], 2)) + ' ' + str(round(own_tot[1] * gold[i], 2)) +
                        #           ' | sold: day ' + str(i) + ' contain: ' + str(round(gold_t[0], 2)) +
                        #           ' in_value: ' + str(round(gold_t[1], 2)) + ' now: ' + str(gold[i]) +
                        #           ' earn:' + str(round(gold[i] * gold_t[0] - gold_t[1] * gold_t[0] / ALPHA11, 2))
                        #           + '\n')
            use_money = max(min(res * b1, CONTAIN * tot_money - own_tot[0] * bitb[i]), 0)
            res -= use_money
            get_tr = use_money / bitb[i] * ALPHA1 * ALPHA1
            lis_own_b.append([get_tr, bitb[i], bitb[i]])
            own_tot[0] += get_tr
            parc_ck += 1
            # f_k.write(str(round(own_tot[0], 2)) + ' ' + str(round(own_tot[0] * bitb[i], 2)) +
            #           ' | parc: day ' + str(i) +
            #           ' contain: ' + str(round(get_tr, 2)) +
            #           ' value: ' + str(bitb[i]) + '\n')
        elif opt_k and opt_g:
            use_money = max(min(res * b1, CONTAIN * tot_money - own_tot[0] * bitb[i]), 0)
            res -= use_money
            get_tr = use_money / bitb[i] * ALPHA1 * ALPHA1
            lis_own_b.append([get_tr, bitb[i], bitb[i]])
            own_tot[0] += get_tr
            # f_k.write(str(round(own_tot[0], 2)) + ' ' + str(round(own_tot[0] * bitb[i], 2)) +
            #           ' | parc: day ' + str(i) +
            #           ' contain: ' + str(round(get_tr, 2)) +
            #           ' value: ' + str(bitb[i]) + '\n')
            use_money = max(min(res * b2, CONTAIN * tot_money - own_tot[1] * gold[i]), 0)
            res -= use_money
            get_tr = use_money / gold[i] * ALPHA2 * ALPHA2
            lis_own_g.append([get_tr, gold[i], gold[i]])
            own_tot[1] += get_tr
            parc_cg += 1
            parc_ck += 1
            # f_g.write(str(round(own_tot[1], 2)) + ' ' + str(round(own_tot[1] * gold[i], 2)) +
            #           ' | parc: day ' + str(i) +
            #           ' contain: ' + str(round(get_tr, 2)) +
            #           ' value: ' + str(gold[i]) + '\n')

        tot_money = res + own_tot[0] * bitb[i] + own_tot[1] * gold[i]
        print(i, ': ', tot_money, '    | ', own_tot[0] * bitb[i], own_tot[1] * gold[i])
        f_t.write(str(round(tot_money, 2)) + ' ' + str(round(own_tot[0] * bitb[i], 2)) + ' '
                  + str(round(own_tot[1] * gold[i], 2)) + '\n')
        # f_t.write(str(round(tot_money, 2)) + '\n')

    # print(tot_money)
    # print('gold parchase: ', parc_cg)
    # print('bitb parchase: ', parc_ck)
    # print('tot : ', parc_cg + parc_ck)
    # print('gold sold : ', sold_cg)
    # print('gold sold1 : ', sold_cg1)
    # print('gold sold2 : ', sold_cg2)
    # print('bitb sold : ', sold_ck)
    # print('bitb sold1 : ', sold_ck1)
    # print('bitb sold2 : ', sold_ck2)
    # sold_gt = sold_cg + sold_cg1 + sold_cg2
    # print('gold sold tot: ', sold_gt)
    # sold_kt = sold_ck + sold_ck1 + sold_ck2
    # print('bitb sold tot: ', sold_kt)
    # print('tot : ', sold_kt + sold_gt)

"""
原
gold parchase:  130
bitb parchase:  393
tot :  523
gold sold (正常卖出):  0
gold sold1 (止损卖出):  74
gold sold2 (投资比例优化卖出):  101
bitb sold (正常卖出):  0
bitb sold1 (止损卖出):  354
bitb sold2 (投资比例优化卖出):  77
gold sold tot:  175
bitb sold tot:  431
tot :  606

佣金为3% 2%
将止损策略调整为 83 87
gold parchase:  123
bitb parchase:  377
tot :  500
gold sold :  0
gold sold1 :  55
gold sold2 :  85
bitb sold :  0
bitb sold1 :  310
bitb sold2 :  70
gold sold tot:  140
bitb sold tot:  380
tot :  520


佣金为4% 3%
将止损策略调整为 78 84
gold parchase:  78
bitb parchase:  360
tot :  438
gold sold :  0
gold sold1 :  25
gold sold2 :  88
bitb sold :  0
bitb sold1 :  304
bitb sold2 :  39
gold sold tot:  113
bitb sold tot:  343
tot :  456


佣金为1% 0.5%
将止损策略调整为 87 91
gold parchase:  125
bitb parchase:  420
tot :  545
gold sold :  0
gold sold1 :  69
gold sold2 :  111
bitb sold :  0
bitb sold1 :  388
bitb sold2 :  74
gold sold tot:  180
bitb sold tot:  462
tot :  642

Process finished with exit code 0

"""