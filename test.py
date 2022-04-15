import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import csv
import numpy as np
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import random
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score

class PSO(object):
    def __init__(self, number_to_pso, max_iteration,
                 all_of_predit_y, y_ture, n_particles, W=1, c1=2, c2=2):
        self.number_to_pso = number_to_pso
        self.max_iteration = max_iteration
        self.y_ture = y_ture
        self.all_of_predit_y = all_of_predit_y
        self.n_particles = n_particles
        self.W = W
        self.c1 = c1
        self.c2 = c2
        self.particle_position_vector = np.random.randn(self.number_to_pso, self.n_particles)
        self.pbest_position = self.particle_position_vector
        self.pbest_fitness_value = np.ones(self.n_particles) * 10000
        self.gbest_fitness_value = 10000
        self.gbest_position = np.zeros(self.number_to_pso)
        self.velocity_vector = np.zeros([number_to_pso, n_particles + 1])
        # self.velocity_vector = ([np.zeros(self.number_to_pso) for _ in range(self.n_particles + 1)])

    def fitness_function(self, weights):
        # 计算加权平均后的y值
        all_of_predit_y = np.sum(weights * self.all_of_predit_y, axis=1)
        # rmse = []
        # mae = []
        # r2 = []
        # huiguifangcha = []
        # for i in range(0, len(all_of_predit_y)):
        #     rmse.append(mean_squared_error(self.y_ture[:,i], all_of_predit_y[:,i]))# 均方差
        #     mae.append(mean_absolute_error(self.y_ture[:,i], all_of_predit_y[:,i])) # 平均绝对误差
        #     r2.append(r2_score(self.y[:,i], all_of_predit_y[:,i]))  # R2
        #     huiguifangcha.append(explained_variance_score(self.y[:,i], all_of_predit_y[:,i]))  #回归方差(反应自变量与因变量之间的相关程度)
        # score = []
        rmse = mean_squared_error(self.y_ture, all_of_predit_y)  # 均方差
        mae = mean_absolute_error(self.y_ture, all_of_predit_y) # 平均绝对误差
        # r2 = r2_score(self.y_ture, all_of_predit_y)  # R2
        huiguifangcha = explained_variance_score(self.y_ture, all_of_predit_y)  #回归方差(反应自变量与因变量之间的相关程度)
        loss = 0.5 * rmse + np.sqrt(0.5) * mae  + 2 / huiguifangcha
        return loss


    def optimization_update(self):
        iteration = 0
        while iteration < self.max_iteration:
            # plot(particle_position_vector)
            for i in range(self.n_particles):
                fitness_cadidate = self.fitness_function(weights=self.particle_position_vector[:, i])
                print("loss in optimization-", i, "is ", fitness_cadidate, " At weights in: ",
                      self.particle_position_vector[:, i])

                if (self.pbest_fitness_value[i] > fitness_cadidate):
                    self.pbest_fitness_value[i] = fitness_cadidate
                    self.pbest_position[:, i] = self.particle_position_vector[:, i]

                if (self.gbest_fitness_value > fitness_cadidate):
                    self.gbest_fitness_value = fitness_cadidate
                    self.gbest_position = self.particle_position_vector[:, i]
                elif (self.gbest_fitness_value == fitness_cadidate and self.gbest_fitness_value > fitness_cadidate):
                    self.gbest_fitness_value = fitness_cadidate
                    self.gbest_position = self.particle_position_vector[:, i]

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
        return self.gbest_position


if __name__ == "__main__":

    # all_of_predit_y = np.array([[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4]])
    # y_ture = all_of_predit_y[:, 1] - 0.5
    # n_particles = 100
    # array_of_y = all_of_predit_y
    # array_of_y_main = (array_of_y - np.min(array_of_y)) / (np.max(array_of_y) - np.min(array_of_y))
    #
    # # opt = PSO(number_to_pso=all_of_predit_y.shape[1], max_iteration=1000,
    # #           all_of_predit_y=all_of_predit_y, y_ture=y_ture, n_particles=n_particles)
    # # best_weight = opt.optimization_update()
    #
    # print(array_of_y)

    # print(np.sum(a * b, axis=1))

    data = {'多元回归': ([[0.32879082],
                    [0.22624374],
                    [0.72041393],
                    [0.72041393],
                    [0.16235619]]),
            '岭回归': ([[0.32879113],
                        [0.22608436],
                        [0.72041366],
                    [0.72041366],
                        [0.16226866]]),
            '贝叶斯回归': ([[0.32879092],
                            [0.22615454],
                            [0.72041385],
                            [0.72041385],
                            [0.16235637]]),
     '核支持向量机': ([[0.31096882],
                      [0.13277762],
                      [0.7355688],
                      [0.73815927],
                      [0.06061371]]), '熵权法': ([[0.31687141],
                                                    [0.19067154],
                                                    [0.73054955],
                                                    [0.73083941],
                                                    [0.1273639]]), '粒子群': ([[112.78670687],
                                                                                 [10.58932265],
                                                                                 [-94.90776347],
                                                                                 [0.35317338],
                                                                                 [10.55497048]])}

    data = np.array(list(data.values())).reshape(6,-1)
    print(data)

