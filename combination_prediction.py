#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import math
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score, median_absolute_error
from sklearn import linear_model, tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
from PSO_and_ACO_Optimization import PSO, ACO
from lightgbm.sklearn import LGBMRegressor
from time import *
from tools import *


class Prediction_model(object):

    def __init__(self, x_data, y_data,n_particles=None, max_iteration=None, prediciction_class="bicha"):
        self.x_data = x_data
        self.y_data = y_data
        self.n_particles = n_particles
        self.max_iteration = max_iteration
        self.prediciction_class = prediciction_class
        self.C1 = 11.68
        self.C2 =67.47
        self.TC1 = 1
        self.TC2 = 2
        self.tan1 = 0.8
        self.tan2 = 0.4
        self.X = 2.2 * pow(10, 12)
        self.N = 39515 / 36
        self.S = 30
        self.U = 110 / math.sqrt(3) * 1000
        self.cosA = 0.8
        self.sinA = 0.6
        self.error = {}
        self.used_time = {}

    def zonghe_error(self, input_y):
        error_of_all_of_y = []
        error_of_y = []

        for y in input_y.T:
            error_of_y.append(float(np.sqrt(mean_squared_error(y_true=self.y_data, y_pred=y))))        #  RMSE
            error_of_y.append(mean_absolute_error(self.y_data, y))                                     #  MAE
            error_of_y.append(float(np.sum(abs((self.y_data - y) / self.y_data)) / len(self.y_data)))  #  mAPE
            error_of_y.append(mean_squared_error(self.y_data, y))                                      #  MSE
            # error_of_y.append(median_absolute_error(self.y_data, y))
            error_of_all_of_y.append(error_of_y)
            error_of_y = []
        error_of_all_of_y = np.array(error_of_all_of_y).reshape(-1, input_y.shape[1])
        return error_of_all_of_y

    def Multiple_Regression(self):
        print("*******************开始多元回归**********************")
        mlr = linear_model.LinearRegression()
        mlr.fit(self.x_data, self.y_data)
        y_test = mlr.predict(self.x_data)
        train_inf = np.isinf(y_test)
        y_test[train_inf] = 100
        self.error['多元回归'] = self.zonghe_error(y_test)
        print("*******************结束**********************")
        return y_test

    def ridge_regression(self):
        print("*******************岭回归**********************")
        mlr = linear_model.RidgeCV(alphas=[0.1, 1, 10])
        mlr.fit(self.x_data, self.y_data)
        y_test = mlr.predict(self.x_data)
        self.error['岭回归'] = self.zonghe_error(y_test)
        print("*******************结束**********************")
        return y_test

    def GBDT_regression(self):
        print("*****************  GBDT回归   *********************")
        mlr = GradientBoostingRegressor()
        mlr.fit(self.x_data, self.y_data)
        start = time()
        y_test = mlr.predict(self.x_data).reshape(-1,1)
        end = time()
        self.used_time['GBDT'] = end - start
        self.error['GBDT回归'] = self.zonghe_error(y_test)
        print("******************   结 束  *********************")
        return y_test

    def LightGBM_regression(self):
        print("*****************  LightGBM回归   *********************")
        mlr = LGBMRegressor()
        mlr.fit(self.x_data, self.y_data)
        start_time = time()
        y_test = mlr.predict(self.x_data).reshape(-1, 1)
        end_time = time()
        self.used_time['LightGBM'] = end_time - start_time
        self.error['LightGBM回归'] = self.zonghe_error(y_test)
        print("******************   结 束  *********************")
        return y_test


    def Random_Forest(self):
        print("*******************贝叶斯回归**********************")
        mlr = linear_model.BayesianRidge()
        # mlr = RandomForestRegressor(n_estimators=20)
        mlr.fit(self.x_data, self.y_data)
        y_test = mlr.predict(self.x_data).reshape(-1, 1)
        self.error['贝叶斯回归'] = self.zonghe_error(y_test)
        print("*******************结束**********************")
        return y_test

    def svm_Ressgression(self, C=10 ** 2.5, kernel='rbf', gamma=10 ** -3.5):
        print("*******************核支持向量机**********************")
        mlr = SVR(C=C, kernel=kernel, gamma=gamma)
        mlr.fit(self.x_data, self.y_data)
        start_time = time()
        y_test = mlr.predict(self.x_data).reshape(-1, 1)
        end_time = time()
        self.used_time['核支持向量机'] = end_time - start_time
        self.error['核支持向量机'] = self.zonghe_error(y_test)
        print("*******************结 束**********************")
        return y_test

    def temperature_disturb(self):
        dertaT = self.x_data[:, 0] - 20
        if self.prediciction_class == "bicha":
            temperature_bicha = self.C1 / (self.C1 + self.C2) - self.C1 * (1 + self.TC1 * dertaT) / (
                    self.C1 * (1 + self.TC1 * dertaT) + self.C2 * (1 + self.TC2 * dertaT))
            return temperature_bicha
        elif self.prediciction_class == "jiaocha":
            temperature_jiaocha = (self.tan2 - self.tan1) * (
                    self.C2 * (1 + self.TC2 * dertaT) / (self.C1 * (1 + self.TC1 * dertaT) + self.C2 * (1 + self.TC2 * dertaT)) - self.C2 / (
                        self.C1 + self.C2)) * 3438
            temperature_jiaocha = temperature_jiaocha % 360
            return temperature_jiaocha


    def humidity_disturb(self):
        R = self.x_data[:,1]
        if self.prediciction_class == "bicha":
            shidu_bicha = 0.17 - ((R * self.cosA) + (self.X * self.sinA)) / (
                    (pow(self.N, 2) * pow(self.U, 2) / (self.S * self.cosA)) + (pow(self.N, 2) * pow(self.U, 2) / (self.S * self.sinA)))
            return shidu_bicha
        elif self.prediciction_class == "jiaocha":
            shidu_jiaocha = -0.4 + ((R * self.cosA) - (self.X * self.sinA)) / (
                    (pow(self.N, 2) * pow(self.U, 2) / (self.S * self.cosA)) + (pow(self.N, 2) * pow(self.U, 2) / (self.S * self.sinA))) * 3438
            return shidu_jiaocha
        # shidu_jiaocha = (shidu_jiaocha,360)


    def electricity_disturb(self):

        C1 = 11.88
        C2 = 67.47
        LK = 129.45
        RK = 1.82
        LM = 2103.64
        RM = 1241
        RF = 131.8
        LF = 639.576
        rf = 6
        CF = 15.868
        w = 50
        S = self.x_data[:,2]
        M = 1 / ((np.sqrt(50 * 5.09) / 2133.33) + 1)
        F = np.zeros([1,len(S)])
        P = np.zeros([1, len(S)])
        for i in range(len(S)):
            W = 2133.33 * 1.25 / M
            RD = W * M / S[i]
            LD = W * np.sqrt(1 - M ** 2) / w / S[i]

            A = (w * RM * LM) * 1j / (RM + 1j * w * LM)
            B = (RF + 1j * w * RF) / (1j * w * CF) / (RF + 1j * w * LF+1 / (1j * w * CF))+RF
            C = (RD + LD)
            D = (A * B * C) / (A * B + A * C + B * C) + RK
            R0 = np.real(D)
            L0 = np.imag(D)
            if self.prediciction_class == "bicha":
                F[0,i] = (np.sqrt(((w ** 2 * C1 * L0) ** 2) + (w * C1 * R0) ** 2) / np.sqrt((w ** 2 * (LK + L0) * (C1 + C2) - 1) ** 2 + (w * (C1 + C2) * R0) ** 2) - C1 / (C1 + C2)) / C1 * (C1 + C2)
            elif self.prediciction_class == "jiaocha":
                P[0,i] = math.atan(-R0 / w / L0) - math.atan(-w * (C1 + C2) * R0) / (w ** 2 * (LK + L0) * (C1 + C2) - 1)

        if self.prediciction_class == "bicha":
            return F
        elif self.prediciction_class == "jiaocha":
            return P


    def shangquanfa_weight_use_y(self, array_of_y):
        '''
        【模型1， 模型2，。。。，模型N】
        返回1*N的数组
        :param array_of_y: M * N
        :return: 1 * N
        '''
        array_of_y = (array_of_y - np.min(array_of_y)) / (np.max(array_of_y) - np.min(array_of_y))
        m = array_of_y.shape[0]
        n = array_of_y.shape[1]
        k = 1 / np.log(m)
        yij = array_of_y.sum(axis=0)  # axis=0列相加 axis=1行相加
        pij = array_of_y / yij
        test = pij * np.log(pij)
        test = np.nan_to_num(test)
        # 将nan空值转换为0
        ej = -k * (test.sum(axis=0))
        # 计算每种指标的信息熵
        wi = (1 - ej) / np.sum(1 - ej)
        # 计算每种指标的权重
        return wi

    def shangquanfa_weight_use_error(self, array_of_y):
        '''
        【模型1， 模型2，。。。，模型N】
        返回1*N的数组
        :param array_of_y: M * N
        :return: 1 * N
        '''
        error_of_all_of_y = []
        error_of_y = []
        for y in array_of_y.T:
            error_of_y.append(mean_squared_error(self.y_data, y))
            error_of_y.append(mean_absolute_error(self.y_data, y))
            error_of_y.append(r2_score(self.y_data, y))
            error_of_y.append(explained_variance_score(self.y_data, y))
            error_of_y.append(median_absolute_error(self.y_data, y))
            error_of_all_of_y.append(error_of_y)
            error_of_y = []
        error_of_all_of_y = np.array(error_of_all_of_y).reshape(-1, array_of_y.shape[1])
        error_of_all_of_y = (error_of_all_of_y - np.min(error_of_all_of_y)) / (np.max(error_of_all_of_y) - np.min(error_of_all_of_y))
        m = error_of_all_of_y.shape[0]
        n = error_of_all_of_y.shape[1]
        k = 1 / np.log(m)
        yij = error_of_all_of_y.sum(axis=0)  # axis=0列相加 axis=1行相加
        pij = error_of_all_of_y / yij
        test = pij * np.log(pij)
        test = np.nan_to_num(test)
        # 将nan空值转换为0
        ej = -k * (test.sum(axis=0))
        # 计算每种指标的信息熵
        wi = (1 - ej) / np.sum(1 - ej)
        # 计算每种指标的权重
        return wi


    def xinxishang(self, array_of_score):
        # array_of_score = (array_of_score - array_of_score.min()) / (array_of_score.max() - array_of_score.min())
        pass

    def pso_weight(self,array_of_y, flag):
        n_particles = 1 # 粒子种群数
        opt = PSO(number_to_pso=array_of_y.shape[1], max_iteration=self.max_iteration,
                  all_of_predit_y=array_of_y, y_ture=self.y_data,
                  n_particles=self.n_particles,opti_weight_flag=flag)
        best_weight = opt.optimization_update()
        return best_weight

    def aco_weight(self,array_of_y):
        n_particles = 1  # 粒子种群数
        opt = ACO(nunber_of_epoch=100, number_of_ants=100,number_of_vars=array_of_y.shape[1],var_num_min=-100, var_num_max=100,
                 y_ture=self.y_data, all_of_predit_y=array_of_y)
        best_weight = opt.main()
        return best_weight


    def prediction_one_by_one(self):
        y_linear = self.Multiple_Regression()
        # y_ridge = self.ridge_regression().reshape(-1, 1)
        y_random_forest = self.Random_Forest()
        y_GBDT = self.GBDT_regression()
        y_LightGBM = self.LightGBM_regression()
        y_svr = self.svm_Ressgression()
        y_temperature = self.temperature_disturb().reshape(-1, 1)
        y_humidity = self.humidity_disturb().reshape(-1, 1)
        y_ele = self.electricity_disturb().reshape(-1, 1)


        all_of_y = np.array(y_linear).reshape(-1, 1)
        all_of_y = np.append(all_of_y, y_random_forest, axis=1)
        all_of_y = np.append(all_of_y, y_GBDT, axis=1)
        all_of_y = np.append(all_of_y, y_LightGBM, axis=1)
        all_of_y = np.append(all_of_y, y_svr, axis=1)
        all_of_y = np.append(all_of_y, y_temperature, axis=1)
        all_of_y = np.append(all_of_y, y_humidity, axis=1)
        all_of_y = np.append(all_of_y, y_ele, axis=1)

        save_y = np.append(all_of_y, self.y_data, axis=1)
        save_csv(csv_file_name='所有Y的值', save_data=save_y)


        for y in all_of_y:
            train_inf = np.isnan(y)
            y[train_inf] = 0

        return all_of_y



    def calculate_weights_of_regression_models(self, all_of_y):
        '''
        用熵权法、信息熵法、蚁群（粒子群）算法
        求取每个回归模型
        在加权求和时候的权重
        :return:
        '''
        ## 熵权法确定权重
        start_time = time()
        print("*******************熵权法确定权重**********************")
        w_of_shangquan = self.shangquanfa_weight_use_error(all_of_y).reshape(-1, 1)
        end_time = time()
        self.used_time["熵权法用时"] = end_time - start_time
        # y_of_shangquan = np.multiply(all_of_y, w_of_shangquan)
        ## PSO确定权重
        print("*******************基础PSO确定权重**********************")
        # w_of_pso = self.pso_weight(all_of_y).reshape(-1, 1)
        start_time = time()
        w_of_pso = self.pso_weight(all_of_y, flag='fixed').reshape(-1, 1)
        end_time = time()
        self.used_time["基础PSO用时"] = end_time - start_time

        print("*******************权重衰减PSO确定权重**********************")
        # w_of_pso = self.pso_weight(all_of_y).reshape(-1, 1)
        start_time = time()
        w_of_pso_decay = self.pso_weight(all_of_y, flag='decay').reshape(-1, 1)
        end_time = time()
        self.used_time["权重衰减PSO用时"] = end_time - start_time

        print("*******************雷--权重衰减PSO确定权重**********************")
        # w_of_pso = self.pso_weight(all_of_y).reshape(-1, 1)
        start_time = time()
        w_of_pso_lei_decay = self.pso_weight(all_of_y, flag='lei_decay').reshape(-1, 1)
        end_time = time()
        self.used_time["雷--权重衰减PSO用时"]  = end_time - start_time

        # y_of_pso = np.multiply(all_of_y, w_of_pso)
        print("**********************ACO确定权重************************")
        start_time = time()
        w_of_aco = self.aco_weight(all_of_y).reshape(-1, 1)
        end_time = time()
        self.used_time["ACO用时"] = end_time - start_time


        return w_of_shangquan, w_of_pso, w_of_pso_decay, w_of_pso_lei_decay, w_of_aco

    def parallel_perdiction(self):
        all_of_y = self.prediction_one_by_one()

        ## 等权重相加
        mean_w = np.array([1/8, 1/8, 1/8, 1/8, 1/8, 1/8, 1/8, 1/8]).reshape(-1,1)
        # mean_w = np.array([0.5, 0.5]).reshape(-1,1)

        y_of_mean_w = np.dot(all_of_y, mean_w)

        self.error['等权重'] = self.zonghe_error(y_of_mean_w)

        self.w_of_shangquan, self.w_of_pso, self.w_of_pso_decay, \
        self.w_of_pso_lei_decay, self.w_of_aco = self.calculate_weights_of_regression_models(all_of_y)
        ## 熵权法确定y
        y_of_shangquan = np.dot(all_of_y, self.w_of_shangquan)
        self.error['熵权法'] = self.zonghe_error(y_of_shangquan)
        ## PSO确定y
        y_of_pso = np.dot(all_of_y, self.w_of_pso)
        self.error['粒子群'] = self.zonghe_error(y_of_pso)

        y_of_pso_decay = np.dot(all_of_y, self.w_of_pso_decay)
        self.error['粒子群decay'] = self.zonghe_error(y_of_pso_decay)

        y_of_pso_lei_decay = np.dot(all_of_y, self.w_of_pso_lei_decay)
        self.error['粒子群_lei_decay'] = self.zonghe_error(y_of_pso_lei_decay)
        ## ACO确定y
        y_of_aco = np.dot(all_of_y, self.w_of_aco)
        self.error['蚁群'] = self.zonghe_error(y_of_aco)

        return y_of_shangquan, y_of_pso

    def get_error(self):
        return self.error
    def get_shangquan_weights(self):
        return self.w_of_shangquan
    def get_pso_weights(self):
        return self.w_of_pso
    def get_aco_weights(self):
        return self.w_of_aco
    def get_used_time(self):
        return self.used_time