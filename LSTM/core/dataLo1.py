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


class DataLoader():
    """A class for loading and transforming data for the lstm model"""

    def __init__(self, filename, split, cols, seq_len):
        self.seq_len = int(seq_len)
        self.cols = cols
        dataframe = pd.read_csv(filename)
        self.dataframe = dataframe
        i_split = int(len(dataframe) * 0)
        self.data_train = dataframe.get(cols).values[:len(dataframe)]
        self.data_test  = []
        self.len_train  = self.seq_len + 1
        self.len_test   = len(self.data_test)
        self.len_train_windows = None
        self.real_data = self.data_train.copy()

        las = self.real_data[0][0]
        for i in range(self.real_data.shape[0]):
            self.data_train[i][0] = ((self.real_data[i][0] / las) - 1) * 20
            las = self.real_data[i][0]
        self.data_train_o = self.data_train.copy()

    def add(self, step):
        self.len_train += step

    def wave(self, len1):
        # print(len1)
        # print(self.data_train[:len1, 0].shape)
        # print(self.data_train_o[:len1, 0].shape)
        # print(wave_lv(self.data_train_o[:len1, 0]).shape)
        # input()
        self.data_train[:len1, 0] = wave_lv(self.data_train_o[:len1, 0])[:len1]

    def get_test_data(self, seq_len, normalise):
        '''
        Create x, y test data windows
        Warning: batch method, not generative, make sure you have enough memory to
        load data, otherwise reduce size of the training split.
        '''
        data_windows = []
        for i in range(self.len_test - seq_len):
            data_windows.append(self.data_test[i:i+seq_len])
        data_windows = np.array(data_windows).astype(float)
        # print(data_windows.shape)
        data_windows = self.normalise_windows(data_windows, single_window=False) if normalise else data_windows
        # print(data_windows)
        x = data_windows[:, :-1]
        y = data_windows[:, -1, [0]]
        return x, y

    def get_train_data(self, seq_len):
        '''
        Create x, y process data windows
        Warning: batch method, not generative, make sure you have enough memory to
        load data, otherwise use generate_training_window() method.
        '''
        data_x = []
        data_y = []
        for i in range(self.len_train - seq_len):
            x, y = self._next_window(i, seq_len)
            data_x.append(x)
            data_y.append(y)
        return np.array(data_x), np.array(data_y)

    def generate_train_batch(self, seq_len, batch_size):
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
                x, y = self._next_window(i, seq_len)
                x_batch.append(x)
                y_batch.append(y)
                i += 1
            yield np.array(x_batch), np.array(y_batch)

    def _next_window(self, i, seq_len):
        '''Generates the next data window from the given index location i'''
        window = self.data_train[i:i+seq_len]
        x = window[:-1]
        y = window[-1, [0]]
        return x, y

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
        return self._next_window(self.len_train - self.seq_len, self.seq_len)

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

    def get_real(self, lis, idi):
        idi -= self.seq_len
        print(lis.shape)
        [lis] = lis
        for i in range(lis.shape[0]):
            lis[i][0] = (lis[i][0] + 1) * self.real_data[idi]
            idi += 1
        return lis

