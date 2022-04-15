#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score, median_absolute_error
import random
from tools import *


class PSO(object):
    def __init__(self, number_to_pso, max_iteration,
                 all_of_predit_y, y_ture, n_particles,
                 W=1, c1=0.3, c2=0.4, opti_weight_flag='fixed',alfha_lei=0.8, beta_lei=0.8):
        self.number_to_pso = number_to_pso
        self.max_iteration = max_iteration
        self.y_ture = y_ture
        self.all_of_predit_y = all_of_predit_y
        self.n_particles = n_particles
        self.W_max = W
        self.c1 = c1
        self.c2 = c2
        self.opti_weight_flag = opti_weight_flag
        self.alfha_lei = alfha_lei
        self.beta_lei = beta_lei
        self.particle_position_vector = np.random.randn(self.number_to_pso, self.n_particles)
        self.pbest_position = self.particle_position_vector
        self.pbest_fitness_value = np.ones(self.n_particles) * 1000000
        self.gbest_fitness_value = 1000000
        self.gbest_position = np.zeros(self.number_to_pso)
        self.velocity_vector = np.zeros([number_to_pso, n_particles + 1])
        self.W = 0
        # self.velocity_vector = ([np.zeros(self.number_to_pso) for _ in range(self.n_particles + 1)])

    def fitness_function(self, weights):
        # 计算加权平均后的y值
        all_of_predit_y = np.sum(weights * self.all_of_predit_y, axis=1)
        rmse = mean_squared_error(self.y_ture, all_of_predit_y)# 均方差
        mae = mean_absolute_error(self.y_ture, all_of_predit_y) # 平均绝对误差
        # huiguifangcha = explained_variance_score(self.y_ture, all_of_predit_y)  # 回归方差(反应自变量与因变量之间的相关程度)

        # r2 = r2_score(self.y_ture, all_of_predit_y)  # R2
        # huiguifangcha = explained_variance_score(self.y_ture, all_of_predit_y)  #回归方差(反应自变量与因变量之间的相关程度)
        loss = 0.5 * abs(rmse) + 5 * np.sqrt(0.5) * abs(mae)
        return abs(loss)


    def optimization_update(self):
        iteration = 0
        w = []
        each_iter_best_fitness = []
        while iteration < self.max_iteration:
            # plot(particle_position_vector)
            all_particle_fitness = []
            all_particle_position = []
            for i in range(self.n_particles):
                fitness_cadidate = self.fitness_function(weights=self.particle_position_vector[:, i])
                all_particle_fitness.append(fitness_cadidate)
                all_particle_position.append(self.particle_position_vector[:, i])
                print("loss in optimization-", i, "  in  ", iteration, "  is ", fitness_cadidate, " At weights in: ",
                      self.particle_position_vector[:, i])

                if (self.pbest_fitness_value[i] > fitness_cadidate):
                    self.pbest_fitness_value[i] = fitness_cadidate
                    self.pbest_position[:, i] = self.particle_position_vector[:, i]

                if (self.gbest_fitness_value > fitness_cadidate):
                    self.gbest_fitness_value = fitness_cadidate
                    self.gbest_position = self.particle_position_vector[:, i]
                elif (self.gbest_fitness_value == fitness_cadidate):
                # elif (self.gbest_fitness_value == fitness_cadidate and self.gbest_fitness_value > fitness_cadidate):
                    self.gbest_fitness_value = fitness_cadidate
                    self.gbest_position = self.particle_position_vector[:, i]

            ## 权重的改变
            if self.opti_weight_flag == 'fixed':
                self.W = self.W_max
            elif self.opti_weight_flag == 'decay':
                self.W = self.W_max * (1 - iteration / self.max_iteration)
            elif self.opti_weight_flag == 'lei_decay':
                mean_d, list_d = lei_pso_decay_coefficient_mean(all_particle_position, self.gbest_position, flag='D')
                mean_f, list_f = lei_pso_decay_coefficient_mean(all_particle_fitness, self.gbest_fitness_value, flag='F')
                # miu_D = min(list_d) / max(list_d)
                # miu_F = min(list_f) / max(list_f)
                miu_D = mean_d / max(list_d)
                miu_F = mean_f / max(list_f)
                self.W = (self.W_max + self.alfha_lei * miu_D + self.beta_lei * miu_F) * (1 - iteration / self.max_iteration)

            w.append(self.W)
            each_iter_best_fitness.append(self.gbest_fitness_value)

            ## 更新每个粒子的位置
            for i in range(self.n_particles):
                velocity  = self.W * self.velocity_vector[:, i]
                person_step = self.c1 * random.random() * (self.pbest_position[:, i] - self.particle_position_vector[:, i])
                goal_step = (self.c2 * random.random()) * (self.gbest_position - self.particle_position_vector[:, i])
                new_velocity = velocity + person_step + goal_step
                new_position = new_velocity + self.particle_position_vector[:, i]
                self.particle_position_vector[:, i] = new_position
                self.velocity_vector[:, i+1] = velocity

            iteration = iteration + 1

        print('pbest_position:', self.pbest_position)
        print('gbest_position:', self.gbest_position)

        save_csv(csv_file_name=self.opti_weight_flag + '__适应度变化', save_data=each_iter_best_fitness)
        save_csv(csv_file_name=self.opti_weight_flag + '__权重变化', save_data=w)

        return self.gbest_position


class ACO(object):
    def __init__(self, nunber_of_epoch=100, number_of_ants=100,
                 number_of_vars=5,var_num_min=-100, var_num_max=100,
                 y_ture=None, all_of_predit_y=None):
        """
        Ant Colony Optimization
        parameter: a list type, like [NGEN, pop_size, var_num_min, var_num_max]
        """
        # 初始化
        self.NGEN = nunber_of_epoch  # 迭代的代数
        self.pop_size = number_of_ants  # 种群大小
        self.var_num = number_of_vars  # 变量个数
        self.bound = []  # 变量的约束范围
        self.var_num_min = var_num_min
        self.var_num_max = var_num_max
        self.y_ture = y_ture
        self.all_of_predit_y = all_of_predit_y

        self.pop_x = np.zeros((self.pop_size, self.var_num))  # 所有蚂蚁的位置
        self.g_best = np.zeros((1, self.var_num))  # 全局蚂蚁最优的位置

        # 初始化第0代初始全局最优解
        temp = -1
        for i in range(self.pop_size):
            for j in range(self.var_num):
                self.pop_x[i][j] = np.random.uniform(self.var_num_min, self.var_num_max)
            fit = self.fitness_function(self.pop_x[i])
            if fit > temp:
                self.g_best = self.pop_x[i]
                temp = fit

    def fitness_function(self, weights):
        # 计算加权平均后的y值
        all_of_predit_y = np.sum(weights * self.all_of_predit_y, axis=1)
        rmse = mean_squared_error(self.y_ture, all_of_predit_y)# 均方差
        mae = mean_absolute_error(self.y_ture, all_of_predit_y) # 平均绝对误差
        loss = 0.5 * abs(rmse) + 5 * np.sqrt(0.5) * abs(mae)
        return abs(loss)

    def update_operator(self, gen, t, t_max):
        """
        更新算子：根据概率更新下一时刻的位置
        """
        rou = 0.8  # 信息素挥发系数
        Q = 1  # 信息释放总量
        lamda = 1 / gen
        pi = np.zeros(self.pop_size)
        for i in range(self.pop_size):
            for j in range(self.var_num):
                pi[i] = (t_max - t[i]) / t_max
                # 更新位置
                if pi[i] < np.random.uniform(0, 1):
                    self.pop_x[i][j] = self.pop_x[i][j] + np.random.uniform(-1, 1) * lamda
                else:
                    self.pop_x[i][j] = self.pop_x[i][j] + np.random.uniform(-1, 1) * (
                            self.var_num_max - self.var_num_min) / 2
                # 越界保护
                if self.pop_x[i][j] < self.var_num_min:
                    self.pop_x[i][j] = self.var_num_min
                if self.pop_x[i][j] > self.var_num_max:
                    self.pop_x[i][j] = self.var_num_max
            # 更新t值
            t[i] = (1 - rou) * t[i] + Q * self.fitness_function(self.pop_x[i])
            # 更新全局最优值
            if self.fitness_function(self.pop_x[i]) > self.fitness_function(self.g_best):
                self.g_best = self.pop_x[i]
        t_max = np.max(t)
        return t_max, t

    def main(self):
        popobj = []
        best = np.zeros((1, self.var_num))[0]
        for gen in range(1, self.NGEN + 1):
            if gen == 1:
                tmax, t = self.update_operator(gen, np.array(list(map(self.fitness_function, self.pop_x))),
                                               np.max(np.array(list(map(self.fitness_function, self.pop_x)))))
            else:
                tmax, t = self.update_operator(gen, t, tmax)
            popobj.append(self.fitness_function(self.g_best))
            print('############ Generation {} ############'.format(str(gen)))
            print(self.g_best)
            print(self.fitness_function(self.g_best))
            if self.fitness_function(self.g_best) > self.fitness_function(best):
                best = self.g_best.copy()
            print('最好的位置：{}'.format(best))
            print('最大的函数值：{}'.format(self.fitness_function(best)))
        print("---- End of ACO (successful) Searching ----")
        return best

