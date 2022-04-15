#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import math
import logging
from logging.handlers import RotatingFileHandler
import sys
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score, median_absolute_error


class two_output_LSTM_Data(object):
    def __init__(self, config):
        self.config = config
        self.x_data, self.y_data = self.read_data()
    def read_data(self):                # 读取初始数据
        df = pd.read_excel(self.config.file_name, self.config.sheet_name)
        df.dropna(inplace=True)
        data_in_file = df.values
        x_data = data_in_file[:, 0:-2]
        bicha_data = data_in_file[:, -2].reshape(-1, 1)
        jiaocha_data = data_in_file[:, -1].reshape(-1, 1)
        mn_x_data = []
        bicha_y = []
        jiaocha_y = []
        max_length = int(len(list(bicha_data)) / self.config.history_times) - self.config.history_times
        for i in range(max_length):
            mn_x_data.append(x_data[i: i + self.config.history_times, :])
            bicha_y.append(bicha_data[i + self.config.history_times])
            jiaocha_y.append(jiaocha_data[i + self.config.history_times])
        mndata = np.array(mn_x_data)
        bicha_y = np.array(bicha_y).reshape([max_length, 1, 1])
        jiaocha_y = np.array(jiaocha_y).reshape([max_length, 1, 1])
        bicha_y = np.append(bicha_y, jiaocha_y, axis=1)
        return mndata, bicha_y

    def get_train_and_valid_data(self, flag=None):
        train_x, valid_x, train_y, valid_y = train_test_split(self.x_data, self.y_data, test_size=self.config.valid_data_rate,
                                                              random_state=self.config.random_seed,
                                                              shuffle=self.config.shuffle_train_data)
        return train_x, valid_x, train_y, valid_y# 划分训练和验证集，并打乱


    def get_all_data(self, flag=None):
        return self.x_data, self.y_data




class LSTM_Data(object):
    def __init__(self, config):
        self.config = config
        self.x_data, self.bicha_data, self.jiaocha_data = self.read_data()
        #
        # self.data_num = self.data.shape[0]
        # self.train_num = int(self.data_num * self.config.train_data_rate)
        #
        # self.mean = np.mean(self.data, axis=0)              # 数据的均值和方差
        # self.std = np.std(self.data, axis=0)
        # self.norm_data = (self.data - self.mean)/self.std   # 归一化，去量纲
        #
        # self.start_num_in_test = 0      # 测试集中前几天的数据会被删掉，因为它不够一个time_step

    def read_data(self):                # 读取初始数据
        df = pd.read_excel(self.config.file_name, self.config.sheet_name)
        df.dropna(inplace=True)
        data_in_file = df.values
        x_data = data_in_file[:, 0:-2]
        bicha_data = data_in_file[:, -2].reshape(-1, 1)
        jiaocha_data = data_in_file[:, -1].reshape(-1, 1)
        mn_x_data = []
        bicha_y = []
        jiaocha_y = []
        max_length = int(len(list(bicha_data)) / self.config.history_times) - self.config.history_times
        for i in range(max_length):
            mn_x_data.append(x_data[i: i + self.config.history_times, :])
            bicha_y.append(bicha_data[i + self.config.history_times])
            jiaocha_y.append(jiaocha_data[i + self.config.history_times])
        mndata = np.array(mn_x_data)
        bicha_y = np.array(bicha_y).reshape([max_length, 1, 1])
        jiaocha_y = np.array(jiaocha_y).reshape([max_length, 1, 1])
        return mndata, bicha_y, jiaocha_y

    def get_train_and_valid_data(self, flag=None):
        # feature_data = self.norm_data[:self.train_num]
        # label_data = self.norm_data[self.config.predict_day : self.config.predict_day + self.train_num,
        #                             self.config.label_in_feature_index]    # 将延后几天的数据作为label
        #
        # if not self.config.do_continue_train:
        #     # 在非连续训练模式下，每time_step行数据会作为一个样本，两个样本错开一行，比如：1-20行，2-21行。。。。
        #     train_x = [feature_data[i:i+self.config.time_step] for i in range(self.train_num-self.config.time_step)]
        #     train_y = [label_data[i:i+self.config.time_step] for i in range(self.train_num-self.config.time_step)]
        # else:
        #     # 在连续训练模式下，每time_step行数据会作为一个样本，两个样本错开time_step行，
        #     # 比如：1-20行，21-40行。。。到数据末尾，然后又是 2-21行，22-41行。。。到数据末尾，……
        #     # 这样才可以把上一个样本的final_state作为下一个样本的init_state，而且不能shuffle
        #     # 目前本项目中仅能在pytorch的RNN系列模型中用
        #     train_x = [feature_data[start_index + i*self.config.time_step : start_index + (i+1)*self.config.time_step]
        #                for start_index in range(self.config.time_step)
        #                for i in range((self.train_num - start_index) // self.config.time_step)]
        #     train_y = [label_data[start_index + i*self.config.time_step : start_index + (i+1)*self.config.time_step]
        #                for start_index in range(self.config.time_step)
        #                for i in range((self.train_num - start_index) // self.config.time_step)]
        #
        # train_x, train_y = np.array(train_x), np.array(train_y)
        if flag == "bicha":
            train_x, valid_x, train_y, valid_y = train_test_split(self.x_data, self.bicha_data, test_size=self.config.valid_data_rate,
                                                              random_state=self.config.random_seed,
                                                              shuffle=self.config.shuffle_train_data)
            return train_x, valid_x, train_y, valid_y# 划分训练和验证集，并打乱
        elif flag == "jiaocha":
            train_x, valid_x, train_y, valid_y = train_test_split(self.x_data, self.jiaocha_data,
                                                                  test_size=self.config.valid_data_rate,
                                                                  random_state=self.config.random_seed,
                                                                  shuffle=self.config.shuffle_train_data)
            return train_x, valid_x, train_y, valid_y
        else:
            print("请选择计算角差还是计算比差")


    def get_all_data(self, flag=None):
        if flag == "bicha":
            return self.x_data, self.bicha_data
        elif flag == "jiaocha":
            return self.x_data, self.jiaocha_data

    # def get_test_data(self, return_label_data=False):
    #     feature_data = self.norm_data[self.train_num:]
    #     sample_interval = min(feature_data.shape[0], self.config.time_step)     # 防止time_step大于测试集数量
    #     self.start_num_in_test = feature_data.shape[0] % sample_interval  # 这些天的数据不够一个sample_interval
    #     time_step_size = feature_data.shape[0] // sample_interval
    #
    #     # 在测试数据中，每time_step行数据会作为一个样本，两个样本错开time_step行
    #     # 比如：1-20行，21-40行。。。到数据末尾。
    #     test_x = [feature_data[self.start_num_in_test+i*sample_interval : self.start_num_in_test+(i+1)*sample_interval]
    #                for i in range(time_step_size)]
    #     if return_label_data:       # 实际应用中的测试集是没有label数据的
    #         label_data = self.norm_data[self.train_num + self.start_num_in_test:, self.config.label_in_feature_index]
    #         return np.array(test_x), label_data
    #     return np.array(test_x)





def four_errors_of_model_result(label_y, predict_y):
    four_errors = {}
    four_errors['RMSE'] = float(np.sqrt(mean_squared_error(y_true=label_y, y_pred=predict_y)))
    four_errors['MAE'] = mean_absolute_error(y_true=label_y, y_pred=predict_y)
    four_errors['mAPE'] = float(np.sum(abs((label_y - predict_y) / label_y)) / len(label_y))
    four_errors['MSE'] = mean_squared_error(y_true=label_y, y_pred=predict_y)
    return four_errors


def read_excel_file(file_name=None, sheet_name=None, flag='nto1',history_times=10):
    df = pd.read_excel(file_name, sheet_name)
    df.dropna(inplace=True)
    data_in_file = df.values
    x_data = data_in_file[:,0:-2]
    bicha_data = data_in_file[:, -2].reshape(-1, 1)
    jiaocha_data = data_in_file[:, -1].reshape(-1, 1)
    if flag == 'nto1':
        return x_data, bicha_data, jiaocha_data
    elif flag == 'mnto1':

        mn_x_data = []
        bicha_y = []
        jiaocha_y = []
        max_length = int(len(list(bicha_data))/history_times)- history_times
        for i in range(max_length):
            mn_x_data.append(x_data[i: i + history_times, :].reshape(-1))
            bicha_y.append(bicha_data[i + history_times])
            jiaocha_y.append(jiaocha_data[i + history_times])
        mndata = np.array(mn_x_data)
        bicha_y = np.array(bicha_y)
        jiaocha_y = np.array(jiaocha_y)
        return mndata, bicha_y, jiaocha_y


def save_csv(csv_file_name, save_data):
    save_temp_data = pd.DataFrame(data=save_data)
    save_temp_data_name = './data/'+ csv_file_name + '.csv'
    save_temp_data.to_csv(save_temp_data_name, encoding='gbk')

def save_data_as_csv(file_name, save_data, string=''):

    save_temp_data = pd.DataFrame(data=save_data)
    work_name = file_name.split('/')[2].split('.')[0]
    temp = np.mat(save_temp_data)
    save_temp_data_name = './data/' + work_name + '_' + string + '.csv'
    save_temp_data.to_csv(save_temp_data_name, encoding='gbk')

def split_dataset(train_data=None, train_target=None):
    X_train, X_test, y_train, y_test = train_test_split(train_data, train_target, test_size=0.4,
                                                                         random_state=0)
    return X_train, X_test, y_train, y_test

def lei_pso_decay_coefficient_mean(all_particle_xx=None, best_xx=None,flag=None):
    mean_value = 0
    distance = []
    if flag == "D":
        for i in range(len(all_particle_xx)):
            distance.append(math.sqrt((all_particle_xx[i][0] - best_xx[0])**2 + (all_particle_xx[i][1] - best_xx[1])**2))
            mean_value = mean_value + distance[i]
    elif flag == "F":
        for i in range(len(all_particle_xx)):
            distance.append(all_particle_xx[i] - best_xx)
            mean_value = mean_value + distance[i]
    mean_value = mean_value / len(all_particle_xx)
    return mean_value, distance

def load_logger(config):
    logger = logging.getLogger()
    logger.setLevel(level=logging.DEBUG)

    # StreamHandler
    if config.do_log_print_to_screen:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(level=logging.INFO)
        formatter = logging.Formatter(datefmt='%Y/%m/%d %H:%M:%S',
                                      fmt='[ %(asctime)s ] %(message)s')
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    # FileHandler
    if config.do_log_save_to_file:
        file_handler = RotatingFileHandler(config.log_save_path + "out.log", maxBytes=1024000, backupCount=5)
        file_handler.setLevel(level=logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # 把config信息也记录到log 文件中
        config_dict = {}
        for key in dir(config):
            if not key.startswith("_"):
                config_dict[key] = getattr(config, key)
        config_str = str(config_dict)
        config_list = config_str[1:-1].split(", '")
        config_save_str = "\nConfig:\n" + "\n'".join(config_list)
        logger.info(config_save_str)

    return logger



