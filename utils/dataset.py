import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings

warnings.filterwarnings('ignore')

class Dataset_Forecasting(Dataset):
    def __init__(self, raw_data: pd.DataFrame, border1s: list, border2s: list, flag='train', 
                 size=None, task='S', target='OT', scale=True, timeenc=0, freq='h'):
        """
        最直接的数据加载方式
        :param flag:        determine the ratio of train/val/test 
        :param size:        [seq_len, label_len, pred_len]
        :param task:        S MS M
        :param target:      the univariate target in S/MS
        :param scale:       normalization
        :param timeenc:     time encoding
        :param freq:        the granularity of time encoding
        """
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val', 'pred']
        self.set_type = 0 if flag=='train' else 1 if flag=='val' else 2 if flag=='test' else 3

        self.task = task
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.__read_data__(raw_data, border1s, border2s)

    def __read_data__(self, df_raw, border1s, border2s):
        self.scaler = StandardScaler()
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.task == 'M' or self.task == 'MS':
            df_data = df_raw[df_raw.columns[1:]]
        elif self.task == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            if self.set_type==3:
                self.scaler.fit(df_data.values)
            else:
                train_data = df_data[border1s[0]:border2s[0]]
                self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # time encoding
        if self.set_type == 3:
            temp_stamp = df_raw[['date']][border1:border2]
            temp_stamp['date'] = pd.to_datetime(temp_stamp.date)
            pred_dates = pd.date_range(temp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq)
            df_stamp = pd.DataFrame(columns=['date'])
            df_stamp.date = list(temp_stamp.date.values) + list(pred_dates[1:])
        else:
            df_stamp = df_raw[['date']][border1:border2]
            df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
        if self.set_type==3:
            self.data_x = data[border1:border2-self.pred_len]
            self.data_y = data[border1:border2]
        else:
            self.data_x = data[border1:border2]
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        if self.set_type==3:
            return len(self.data_x) - self.seq_len + 1
        return len(self.data_x) - self.seq_len - self.pred_len + 1


class Dataset_Forecasting_Solar(Dataset):
    def __init__(self, raw_data: pd.DataFrame, border1s: list, border2s: list, flag='train', 
                 size=None, task='M', scale=True):
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val', 'pred']
        self.set_type = 0 if flag=='train' else 1 if flag=='val' else 2 if flag=='test' else 3

        self.task = task
        self.scale = scale

        self.__read_data__(raw_data, border1s, border2s)

    def __read_data__(self, df_raw, border1s, border2s):
        self.scaler = StandardScaler()
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.task == 'M' or self.task == 'MS':
            df_data = df_raw
        elif self.task == 'S':
            df_data = df_raw[df_raw.columns[-1]]

        if self.scale:
            if self.set_type==3:
                self.scaler.fit(df_data.values)
            else:
                train_data = df_data[border1s[0]:border2s[0]]
                self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        if self.set_type==3:
            self.data_x = data[border1:border2-self.pred_len]
            self.data_y = data[border1:border2]
        else:
            self.data_x = data[border1:border2]
            self.data_y = data[border1:border2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = np.zeros((seq_x.shape[0], 1))
        seq_y_mark = np.zeros((seq_y.shape[0], 1))

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        if self.set_type==3:
            return len(self.data_x) - self.seq_len + 1
        return len(self.data_x) - self.seq_len - self.pred_len + 1
