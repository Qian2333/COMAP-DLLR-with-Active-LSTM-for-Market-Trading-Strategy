import math
import numpy as np
import pandas as pd
import pywt


def wave_lv(aa):
    b = aa.copy()
    b = np.array([b]).T
    threshold1 = 0.04
    colen = pywt.wavedec(b[:, 0], 'db8', level=6)
    for ii in range(1, len(colen)):
        colen[ii] = pywt.threshold(colen[ii], threshold1 * max(colen[ii]))
    return pywt.waverec(colen, 'db8')


def average_window(seq, half_len, step):
    ans_seq = []
    i = 0
    while i < seq.shape[0]:
        tmp = sum(seq[i:min(seq.shape[0], i + half_len)])
        ans_seq.append(tmp / min(half_len, seq.shape[0] - i))
        i += step
    return np.array(ans_seq)


def linear_regression(seq):
    seq = np.array(seq)
    seq.reshape(-1, 1)
    from sklearn import linear_model
    lin_re = linear_model.LinearRegression()
    date = np.zeros((seq.shape[0], 1))
    for i in range(seq.shape[0]):
        date[i][0] = i
    lin_re.fit(date, seq)
    return lin_re.coef_, lin_re.intercept_


def juan(aa):
    b = aa
    # print(b.shape)
    data1 = b.copy()
    # print(b.shape[0])
    # input()
    for i in range(b.shape[0]):
        if i == 0: data1[i] = b[i] * 0.5 + b[i + 1] * 0.5
        elif i + 1 == b.shape[0]: data1[i] = b[i] * 0.5 + b[i - 1] * 0.5
        else: data1[i] = b[i] * 0.34 + b[i - 1] * 0.33 + b[i + 1] * 0.33
    # print(aa)
    # print(data1)
    # input()
    return data1


class DataLoader():
    """A class for loading and transforming data for the lstm model"""

    def __init__(self, filename, split, cols, seq_len):
        self.seq_len = int(seq_len)
        self.cols = cols
        dataframe = pd.read_csv(filename)
        self.dataframe = dataframe
        self.data_train = dataframe.get(cols).values[:len(dataframe)]
        self.data_test  = []
        self.len_train  = self.seq_len + 1
        self.len_test   = len(self.data_test)
        self.len_train_windows = None
        self.real_data = self.data_train.copy()

    def wave(self, len1):
        # print(len1)
        # print(self.data_train[:len1, 0].shape)
        # print(self.data_train_o[:len1, 0].shape)
        # print(wave_lv(self.data_train_o[:len1, 0]).shape)
        # input()
        self.data_train[:len1, 0] = wave_lv(self.real_data[:len1, 0])[:len1]

    def juan(self, len1):
        # print(len1)
        # print(self.data_train[:len1, 0].shape)
        # print(self.data_train_o[:len1, 0].shape)
        # print(wave_lv(self.data_train_o[:len1, 0]).shape)
        # input()
        self.data_train[:len1, 0] = juan(self.real_data[:len1, 0])[:len1]

    def add(self, step):
        self.len_train += step

    def get_train_data(self, seq_len, normalise):
        '''
        Create x, y process data windows
        Warning: batch method, not generative, make sure you have enough memory to
        load data, otherwise use generate_training_window() method.
        '''
        data_x = []
        data_y = []
        for i in range(self.len_train - seq_len):
            x, y = self._next_window(i, seq_len, normalise)
            data_x.append(x)
            data_y.append(y)
        return np.array(data_x), np.array(data_y)

    def generate_train_batch(self, seq_len, batch_size, normalise):
        '''Yield a generator of training data from filename on given list of cols split for process/test'''
        i = 0
        while i < (self.len_train - seq_len):
            x_batch = []
            y_batch = []
            for b in range(batch_size):
                if i >= (self.len_train - seq_len):
                    # stop-condition for a smaller final batch if data doesn't divide evenly
                    yield np.array(x_batch), np.array(y_batch)
                    i = 0
                x, y = self._next_window(i, seq_len, normalise)
                x_batch.append(x)
                y_batch.append(y)
                i += 1
            yield np.array(x_batch), np.array(y_batch)

    def _next_window(self, i, seq_len, normalise):
        '''Generates the next data window from the given index location i'''
        window = average_window(self.data_train[i:i+seq_len], 3, 3)
        # print(window)
        window = self.normalise_windows(window, single_window=True)[0] if normalise else window
        x = window[:-2]
        y = window[-3:]
        y, _ = linear_regression(y)
        # print(y)
        # print(y.shape)
        # input()
        return x, y

    def _next_window_train(self, i, seq_len, normalise):
        '''Generates the next data window from the given index location i'''
        window = average_window(self.data_train[i:(i + seq_len)], 3, 3)
        # print(window)
        window = self.normalise_windows(window, single_window=True)[0] if normalise else window
        x = window
        # y = window[-3:]
        # y, _ = linear_regression(y)
        # print(y)
        # print(y.shape)
        # input()
        return x

    def normalise_windows(self, window_data, single_window=False):
        '''Normalise window with a base value of zero'''
        normalised_data = []
        window_data = [window_data] if single_window else window_data
        for window in window_data:
            normalised_window = []
            for col_i in range(window.shape[1]):
                normalised_col = [((float(p) / float(window[0, col_i])) - 1) for p in window[:, col_i]]
                normalised_window.append(normalised_col)
            normalised_window = np.array(normalised_window).T # reshape and transpose array back into original multidimensional format
            normalised_data.append(normalised_window)
        return np.array(normalised_data)

    def normal_back(self, nt, id):
        return self.data_train[id - self.seq_len + 1] * (nt + 1)

    def test_nxt(self):
        # print(self.len_train - self.seq_len + 1)
        # input()
        return self._next_window(self.len_train - int(self.seq_len * 0.8), self.seq_len, normalise=True)

    def normal_back_w(self, x, y, id):
        # print(x.shape)
        # print(y.shape)
        # input()
        p = self.data_train[id - self.seq_len + 1]
        lis = []
        for i in x:
            lis.append(p * (i + 1))
        lis.append(p * (y[0] + 1))
        return lis
