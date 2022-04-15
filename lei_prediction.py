#!/usr/bin/env python 
# -*- coding:utf-8 -*-
#!/usr/bin/env python
# -*- coding:utf-8 -*-
import csv
import math
import numpy as np
from sklearn.model_selection import train_test_split
from combination_prediction import Prediction_model
from tools import *



if __name__ == '__main__':
    file_name = './data/huganqi/all_change.xlsx'
    sheet_name = '所有'
    max_iteration = 1000
    n_particles = 100
    x_data, bicha_data, jiaocha_data = read_excel_file(file_name=file_name, sheet_name=sheet_name,flag='mnto1')
    ## 比差的组合回归
    X_train_bi, X_test_bi, y_train_bi, y_test_bi = train_test_split(x_data, bicha_data, test_size=0.1, random_state=0)
    multi_regression_model_bi = Prediction_model(x_data=X_train_bi, y_data=y_train_bi, n_particles=n_particles, max_iteration=max_iteration)
    y_of_shangquan_bi, y_of_pso_bi = multi_regression_model_bi.parallel_perdiction()
    bicha_error = multi_regression_model_bi.get_error()
    bicha_used_time = multi_regression_model_bi.get_used_time()
    print(bicha_error)

    ## 教cha 的组合回归
    X_train_jiao, X_test_jiao, y_train_jiao, y_test_jiao = train_test_split(x_data, jiaocha_data, test_size=0.1, random_state=0)
    multi_regression_model_jiao = Prediction_model(x_data=X_train_jiao, y_data=y_train_jiao, n_particles=n_particles, max_iteration=max_iteration)
    y_of_shangquan_jiao, y_of_pso_jiao = multi_regression_model_jiao.parallel_perdiction()
    jiaocha_error = multi_regression_model_jiao.get_error()
    jiaocha_used_time = multi_regression_model_jiao.get_used_time()

    print("#########################################################")
    print("##################### bi cha ##############################")
    print('#######比差误差######')
    print(bicha_error)
    print('#######比差用时######')
    print(bicha_used_time)
    save_data_as_csv(file_name=file_name, save_data=np.array(list(bicha_used_time.values())),
                     string='比差回归时间')
    save_data_as_csv(file_name=file_name, save_data=np.array(list(bicha_error.values())).reshape(-1, 4), string='比差回归误差')
    print("####################### jiao cha ##############################")
    print('#######角差误差######')
    print(jiaocha_error)
    print('#######角差用时######')
    print(jiaocha_used_time)
    save_data_as_csv(file_name=file_name, save_data=np.array(list(bicha_used_time.values())),
                     string='角差回归时间')
    save_data_as_csv(file_name=file_name, save_data=np.array(list(jiaocha_error.values())).reshape(-1, 4), string='角差回归误差')

    y_train_bi = np.append(y_train_bi, y_of_pso_bi, axis=1)
    y_train_jiao = np.append(y_train_jiao, y_of_pso_jiao, axis=1)
    save_data_as_csv(file_name=file_name, save_data=y_train_bi, string='比差回归结果')
    save_data_as_csv(file_name=file_name, save_data=y_train_jiao, string='角差回归结果')











