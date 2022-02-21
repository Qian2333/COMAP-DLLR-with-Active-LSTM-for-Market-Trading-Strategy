import numpy as np
import pandas as pd


def load_d(path):
    with open('BCHAIN-MKPRU.csv', 'r') as f:
        text = f.readlines()
        dates = []
        prices = []
        for line in text:
            [date, price] = line.split(',')
            date = date.split('/')
            dates.append(date)
            prices.append(price)
        return dates, prices

if '__main__' == __name__:
    a = pd.read_csv('goldr.csv')
    prices = a.get('1').values
    print(prices)
    op = a.get('2').values
    print(op)
    # Las = prices[0]
    # for i in prices:
    #     if Las:
    #         prices2.append((i / Las) - 1)
    #     Las = i

    prices = np.array([prices, op], dtype=float).T
    print(prices)
    save = pd.DataFrame(prices, columns=['value', 'op'])
    save.to_csv('gold.csv', index=False)
