import pandas as pd
import numpy as np

M = 1826
mat_dis = np.zeros((M, M))
mat_op = np.zeros((M, M))
alpha1 = 0.02
alpha2 = 0.01


if __name__ == '__main__':
    bit = pd.read_csv('test.csv')
    gold = pd.read_csv('gold.csv')
    bit_l = bit.get('value').values
    gold1 = gold.get('value').values
    gold2 = gold.get('op').values
    print(mat_dis.shape)
    for i in range(M):
        for j in range(i, M):
            mat_dis[i][j] = ((1. - alpha1) ** 2) * float(bit_l[j]) / float(bit_l[i])
            if gold2[i] > 0.01 and gold2[j] > 0.01:
                tmp = ((1. - alpha2) ** 2) * float(gold1[j]) / float(gold1[i])
                if tmp > mat_dis[i][j]:
                    mat_op[i][j] = 1
                    mat_dis[i][j] = tmp
            if 1 > mat_dis[i][j]:
                mat_op[i][j] = 2
                mat_dis[i][j] = 1.
    ans = np.zeros((M))
    op = np.zeros((M, 2))
    ans[0] = 1000
    for i in range(M):
        for j in range(i, M):
            if ans[i] * mat_dis[i][j] > ans[j]:
                ans[j] = ans[i] * mat_dis[i][j]
                op[j][0] = i
                op[j][1] = mat_op[i][j]
    idd = M - 1
    lis = []
    while idd > 0:
        # print(op[idd].shape)
        # print(op[idd][0], ' ', op[idd][1])
        # print(idd)
        # input()
        lis.append([int(op[idd][0]), int(idd), int(op[idd][1])])
        idd = int(op[idd][0])
    with open('bestans.txt', 'w') as f:

        f.write(str(ans[-1]) + '\n')
        lis.reverse()
        for item in lis:
            f.write(str(item[0]) + ' ' + str(item[1]) + ' ' + str(item[2]) + '\n')
