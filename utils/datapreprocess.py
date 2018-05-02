"""
datapreprocess.py

전처리를 위한 class 생성

Preprocess
    preprocess: perP_input + split
    read_data:  filepath로부터 pd.DataFrame 생성
    perP_input: read_data + X(feature), Y(label)로 분리
    perP_new: New_data에 대해서 read_data
    split: training + test 7:3으로 분리
scale_fit: StandardScaler(스케일 맞춰줌)를 데이터에 적용해서 scaler 출력
load_info: 외부 csv 파일 불러오기
"""


import os
import sys
from time import time


class Preprocess:
    def __init__(self, random_state=0):
        self.class_n = 1
        self.data_path = '.'
        self.x_offset = 1
        self.verbose = 0
        self.del_index_g = []
        self.del_feature = False
        self.random_state = random_state
        self.group_info = 'feature_info_label'

    def preprocess(self, input_file):
        if self.verbose >= 1:
            print("data loading...")
        start_time = time()
        X, Y = self.perP_input(input_file)
        X_train, X_test, Y_train, Y_test = self.split(X, Y, random_state=self.random_state)
        # feature를 삭제한다면
        if self.del_feature:
            temp = X_train.shape[1]
            X_train, d_names, d_list = self.delete_feature_group(X_train, with_list=True)
            X_test = self.delete_feature_group(X_test)
            if self.verbose >= 1:
                print('삭제된 feature group: ', d_names)
                end_time = time()
                print("train set: %d, test set: %d, # of features: %d (before: %d)"
                      % (X_train.shape[0], X_test.shape[0], X_train.shape[1], temp))
                print("data loaded, 걸린 시간:", end_time - start_time)
            return X_train, X_test, Y_train, Y_test, d_list
        # feature를 안 삭제한다면
        else:
            if self.verbose >= 1:
                end_time = time()
                print("train set: %d, test set: %d, # of features: %d" % (X_train.shape[0], X_test.shape[0], X_train.shape[1]))
                print("data loaded, 걸린 시간:", end_time - start_time)
            return X_train, X_test, Y_train, Y_test

    def read_data(self, filepath):
        import pandas as pd
        ext_csv = '.csv'
        delim_csv = ','
        ext_tsv = '.txt'
        delim_tsv = '\t'
        filename, ext = os.path.splitext(filepath)
        ext = ext.lower()
        if ext == ext_csv:
            sep = delim_csv
            df = pd.read_csv(filepath, sep=sep, header=None)
        elif ext == ext_tsv:
            sep = delim_tsv
            df = pd.read_csv(filepath, sep=sep, header=None)
        else:
            print("Check file format.\n Supported file formats: csv, txt")
            sys.exit(1)
        return df

    def perP_input(self, input_file):
        # data preprocessing 2: extract values and split train and test
        n = self.class_n
        df = self.read_data(filepath=input_file)
        X = df.iloc[:, :-n].values
        Y = df.iloc[:, -n:].values
        del df
        return X, Y

    def perP_new(self, new_data_file):
        # data preprocessing 2: extract values and split train and test
        df = self.read_data(filepath=new_data_file)
        New_data = df.iloc[:, :].values
        del df
        if self.del_feature:
            New_data = self.delete_feature_group(New_data)
        return New_data

    def split(self, X, Y, test_size=0.3, random_state=0):
        if self.verbose >= 1:
            print('train test split: %.1f' % test_size)
        from sklearn.model_selection import train_test_split
        X, X_test, Y, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state, shuffle=True)
        return X, X_test, Y, Y_test

    def delete_feature(self, X, feature_index):
        import numpy as np
        return np.delete(X, feature_index, axis=1)

    def delete_feature_group(self, X, with_list=False):
        import numpy as np
        temp = load_info(self.data_path, self.group_info)
        group_names = temp['group'].values
        label = temp['number'].values
        accum_label = [label[0]]
        for i in range(1, len(label)):
            accum_label.append(label[i] + accum_label[i - 1])
        del_list = []
        del_names = []
        for index in sorted(self.del_index_g)[::-1]:
            if index == 0:
                feature_index = np.arange(1, accum_label[0]+1)
            else:
                feature_index = np.arange(accum_label[index-1]+1, accum_label[index]+1)
            X = self.delete_feature(X, feature_index)
            del_list = np.concatenate((del_list, feature_index), axis=0)
            del_names.append(group_names[index])
        del_names = del_names[::-1]
        del_list = sorted(del_list.astype(int))
        if with_list:
            return X, del_names, del_list
        else:
            return X

    def half(self, X, Y):
        X_one = X[:round(X.shape[0]/2), :]
        Y_one = Y[:round(Y.shape[0]/2), :]
        X_two = X[round(X.shape[0] / 2):, :]
        Y_two = Y[round(Y.shape[0] / 2):, :]
        return X_one, Y_one, X_two, Y_two


def scale_fit(X):
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    sc.fit(X)
    return sc


def load_info(data_path, name='feature_info_label'):
    import pandas as pd
    info = pd.read_csv(os.path.join(data_path, name + '.csv'), sep=',', header=0, engine='python')
    return info

