import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

from utils.tools import StandardScaler

import warnings
warnings.filterwarnings('ignore')

def time_to_seconds(time_str):
    h, m, s = map(int, time_str.split(':'))
    return h * 3600 + m * 60 + s

class Dataset_EP_second(Dataset):
    def __init__(self, root_path, size=None, flag='train', 
                 features='S', scale=True, inverse=False,
                 data_path='0118.csv', target='TMD52'):
        # size [seq_len, label_len, pred_len]
        if size == None:
            self.seq_len = 60*15
            self.label_len = 60
            self.pred_len = 60
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, 
                                          self.data_path))
        
        cols = list(df_raw.columns); cols.remove(self.target); cols.remove('timestamp') # change
        df_raw = df_raw[['timestamp'] + cols + [self.target]]

        num_train = int(len(df_raw)*0.7)
        num_test = int(len(df_raw)*0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train-self.seq_len, len(df_raw)-num_test-self.seq_len]
        border2s = [num_train, num_train+num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]
        
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['timestamp']][border1:border2]
        #data_stamp = df_stamp['timestamp']
        data_stamp = df_stamp['timestamp'].values

        #data_stamp

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        
        self.data_stamp = data_stamp # to do

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = np.concatenate([self.data_x[r_begin:r_begin+self.label_len], self.data_y[r_begin+self.label_len:r_end]], 0)
        else:
            seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        #seq_x = torch.tensor(seq_x, dtype=torch.float32)
        #seq_y = torch.tensor(seq_y, dtype=torch.float32)
        #seq_x_mark = torch.tensor(self.data_stamp[s_begin:s_end].values, dtype=torch.float32)
        #seq_y_mark = torch.tensor(self.data_stamp[r_begin:r_end].values, dtype=torch.float32)

        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1
    
    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    
class Dataset_EP_sub(Dataset):
    def __init__(self, root_path, size=None, flag='train', 
                 features='S', scale=True, inverse=False,
                 data_path='sub60.csv', target='TMD52'):
        # size [seq_len, label_len, pred_len]
        if size == None:
            self.seq_len = 60*15
            self.label_len = 60
            self.pred_len = 60
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, 
                                          self.data_path))
        
        cols = list(df_raw.columns); cols.remove(self.target); cols.remove('timestamp') # change
        df_raw = df_raw[['timestamp'] + cols + [self.target]]

        num_train = int(len(df_raw)*0.7)
        num_test = int(len(df_raw)*0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train-self.seq_len, len(df_raw)-num_test-self.seq_len]
        border2s = [num_train, num_train+num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]
        
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['timestamp']][border1:border2]
        #data_stamp = df_stamp['timestamp']
        data_stamp = df_stamp['timestamp'].values

        #data_stamp

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        
        self.data_stamp = data_stamp # to do

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = np.concatenate([self.data_x[r_begin:r_begin+self.label_len], self.data_y[r_begin+self.label_len:r_end]], 0)
        else:
            seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        #seq_x = torch.tensor(seq_x, dtype=torch.float32)
        #seq_y = torch.tensor(seq_y, dtype=torch.float32)
        #seq_x_mark = torch.tensor(self.data_stamp[s_begin:s_end].values, dtype=torch.float32)
        #seq_y_mark = torch.tensor(self.data_stamp[r_begin:r_end].values, dtype=torch.float32)

        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1
    
    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='0118.csv', target='TMD52',
                 scale=True, inverse=False):
        # size [seq_len, label_len, pred_len]
        if size == None:
            self.seq_len = 60*15
            self.label_len = 60
            self.pred_len = 60
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        assert flag in ['pred']

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        
        cols = list(df_raw.columns); cols.remove(self.target); cols.remove('timestamp')
        df_raw = df_raw[['timestamp'] + cols + [self.target]]

        border1 = len(df_raw) - self.seq_len
        border2 = len(df_raw)

        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        
        tmp_stamp = df_raw[['timestamp']][border1:border2]
        #tmp_stamp['timestamp'] = pd.to_datetime(tmp_stamp.timestamp)
        data_stamp = tmp_stamp['timestamp'].values
        #tmp_stamp['timestamp'] = tmp_stamp['timestamp'].apply(time_to_seconds)
        
        last_time_s = data_stamp['timestamp'].iloc[-1]

        pred_times = [last_time_s + i for i in range(1, self.pred_len + 1)] # number append

        # 将原始 timestamp 列与预测的秒数直接合并
        df_stamp = pd.DataFrame(columns=['timestamp'])
        df_stamp['timestamp'] = list(tmp_stamp['timestamp'].values) + list(pred_times)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        
        self.data_stamp = df_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin+self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin+self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark
        
    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    

