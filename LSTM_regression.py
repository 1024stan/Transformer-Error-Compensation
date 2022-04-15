#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import os
import time
import torch
from torch.nn import Module, LSTM, Linear, Conv1d
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tools import *
from time import *
import torch.nn as nn




class LSTM_Base_Net(Module):
    '''
    pytorch预测模型，包括LSTM时序预测层和Linear回归输出层
    可以根据自己的情况增加模型结构
    '''
    def __init__(self, config):
        super(LSTM_Base_Net, self).__init__()
        # self.lstm = LSTM(input_size=config.input_size, hidden_size=config.hidden_size,
        #                  num_layers=config.lstm_layers, batch_first=True,
        #                  dropout=config.dropout_rate, bidirectional=False)
        self.history_times = config.history_times
        self.batch_size = config.batch_size
        self.use_bidirectional = config.use_bidirectional
        if self.use_bidirectional == True:
            self.lstm = LSTM(input_size=config.input_size, hidden_size=config.hidden_size,
                             num_layers=config.lstm_layers, batch_first=True,
                             dropout=config.dropout_rate, bidirectional=True)
            self.linear_1 = Linear(in_features=config.hidden_size * 2, out_features=config.output_size)
            self.linear_2 = Linear(in_features=config.history_times, out_features=config.output_size)
        else:
            self.lstm = LSTM(input_size=config.input_size, hidden_size=config.hidden_size,
                             num_layers=config.lstm_layers, batch_first=True,
                             dropout=config.dropout_rate, bidirectional=False)
            self.linear_1 = Linear(in_features=config.hidden_size , out_features=config.output_size)
            self.linear_2 = Linear(in_features=config.history_times, out_features=config.output_size)

    def forward(self, x, hidden=None):
        lstm_out, hidden = self.lstm(x, hidden)
        linear_1_out = self.linear_1(lstm_out)
        linear_1_out = linear_1_out.view(linear_1_out.shape[0], linear_1_out.shape[2], linear_1_out.shape[1])
        linear_out = self.linear_2(linear_1_out)
        return linear_out, hidden



class improved_LSTM_solo_Net(Module):
    '''
    pytorch预测模型，包括LSTM时序预测层和Linear回归输出层
    可以根据自己的情况增加模型结构
    '''
    def __init__(self, config):
        super(improved_LSTM_solo_Net, self).__init__()
        # self.lstm = LSTM(input_size=config.input_size, hidden_size=config.hidden_size,
        #                  num_layers=config.lstm_layers, batch_first=True,
        #                  dropout=config.dropout_rate, bidirectional=False)
        self.history_times = config.history_times
        self.batch_size = config.batch_size
        self.use_bidirectional = config.use_bidirectional

        # model
        self.convs = nn.ModuleList([
                nn.Sequential(Conv1d(in_channels=config.input_size,
                                         out_channels=config.con_output_size,
                                         kernel_size=config.kernel_size),
                                  nn.ReLU(),
                                  nn.MaxPool1d(kernel_size=config.kernel_size))
                for i in range(config.num_convs_layers)])
        # self.conv = Conv1d(in_channels=config.input_size,
        #                                   out_channels=config.con_output_size,
        #                                   kernel_size=config.kernel_size)
        # self.re = nn.ReLU()
        # self.pool = nn.MaxPool1d(kernel_size=config.kernel_size)

        if self.use_bidirectional == True:
            self.lstm = LSTM(input_size=config.con_output_size, hidden_size=config.hidden_size,
                             num_layers=config.lstm_layers, batch_first=True,
                             dropout=config.dropout_rate, bidirectional=True)
            self.linear_1 = Linear(in_features=config.hidden_size * 2, out_features=config.output_size)
        else:
            self.lstm = LSTM(input_size=config.con_output_size, hidden_size=config.hidden_size,
                             num_layers=config.lstm_layers, batch_first=True,
                             dropout=config.dropout_rate, bidirectional=False)
            self.linear_1 = Linear(in_features=config.hidden_size , out_features=config.output_size)

        self.linear_2 = Linear(in_features=config.num_convs_layers,
                               out_features=config.output_size)

    def forward(self, x, hidden=None):
        # con_out = self.conv(x.permute(0, 2, 1))
        # re_out = self.re(con_out)
        # pool_out = self.pool(re_out)
        conv_out = [conv(x.permute(0, 2, 1)) for conv in self.convs]
        conv_out = torch.cat(conv_out, dim=2)


        lstm_out, hidden = self.lstm(conv_out.permute(0, 2, 1), hidden)
        linear_1_out = self.linear_1(lstm_out)
        linear_1_out = linear_1_out.view(linear_1_out.shape[0], linear_1_out.shape[2], linear_1_out.shape[1])
        linear_out = self.linear_2(linear_1_out)
        return linear_out, hidden


class multi_out_layer(Module):
    def __init__(self, in_features, out_features, num_fc_layers, history_times):
        super(multi_out_layer, self).__init__()
        self.linears = nn.ModuleList([
                        Linear(in_features=in_features,
                               out_features=in_features)
                        for i in range(num_fc_layers)])
        self.re = nn.ReLU()
        self.linear_last_1 = Linear(in_features=in_features * num_fc_layers, out_features=out_features)
        self.linear_last_2 = Linear(in_features=history_times, out_features=out_features)

    def forward(self, x):
        linears_out = [linear(x) for linear in self.linears]
        linears_out = torch.cat(linears_out, dim=2)
        linear_last_1_out = self.linear_last_1(linears_out)
        linear_last_1_out = linear_last_1_out.view(linear_last_1_out.shape[0], linear_last_1_out.shape[2], linear_last_1_out.shape[1])
        output = self.linear_last_2(linear_last_1_out)
        return output



class improved_LSTM_multi_Net(Module):
    '''
    pytorch预测模型，包括LSTM时序预测层和Linear回归输出层
    可以根据自己的情况增加模型结构
    '''
    def __init__(self, config):
        super(improved_LSTM_multi_Net, self).__init__()
        # self.lstm = LSTM(input_size=config.input_size, hidden_size=config.hidden_size,
        #                  num_layers=config.lstm_layers, batch_first=True,
        #                  dropout=config.dropout_rate, bidirectional=False)
        self.history_times = config.history_times
        self.batch_size = config.batch_size
        self.use_bidirectional = config.use_bidirectional
        # model
        self.convs = nn.ModuleList([
                nn.Sequential(Conv1d(in_channels=config.input_size,
                                         out_channels=config.con_output_size,
                                         kernel_size=config.kernel_size),
                                  nn.ReLU(),
                                  nn.MaxPool1d(kernel_size=config.kernel_size))
                for i in range(config.num_convs_layers)])
        if self.use_bidirectional == True:
            self.lstm = LSTM(input_size=config.con_output_size, hidden_size=config.hidden_size,
                             num_layers=config.lstm_layers, batch_first=True,
                             dropout=config.dropout_rate, bidirectional=True)
            self.linear_1 = Linear(in_features=config.hidden_size * 2, out_features=config.hidden_size)
        else:
            self.lstm = LSTM(input_size=config.con_output_size, hidden_size=config.hidden_size,
                             num_layers=config.lstm_layers, batch_first=True,
                             dropout=config.dropout_rate, bidirectional=False)
            self.linear_1 = Linear(in_features=config.hidden_size , out_features=config.hidden_size)

        self.bicha = multi_out_layer(in_features=config.hidden_size,
                                       out_features=config.output_size,
                                       num_fc_layers=config.num_fc_layers,
                                     history_times=config.history_times)
        self.jiaocha = multi_out_layer(in_features=config.hidden_size,
                                       out_features=config.output_size,
                                       num_fc_layers=config.num_fc_layers,
                                       history_times=config.history_times)

    def forward(self, x, hidden=None):
        # con_out = self.conv(x.permute(0, 2, 1))
        # re_out = self.re(con_out)
        # pool_out = self.pool(re_out)
        conv_out = [conv(x.permute(0, 2, 1)) for conv in self.convs]
        conv_out = torch.cat(conv_out, dim=2)


        lstm_out, hidden = self.lstm(conv_out.permute(0, 2, 1), hidden)
        lstm_out = self.linear_1(lstm_out)
        bicha_out = self.bicha(lstm_out)
        jiaocha_out = self.jiaocha(lstm_out)
        output = torch.cat([bicha_out, jiaocha_out], dim=1)
        return output, hidden


def train(config, logger, train_and_valid_data):
    if config.do_train_visualized:
        import visdom
        vis = visdom.Visdom(env='model_pytorch')

    train_X, train_Y, valid_X, valid_Y = train_and_valid_data
    train_X, train_Y = torch.from_numpy(train_X).float(), torch.from_numpy(train_Y).float()     # 先转为Tensor
    train_loader = DataLoader(TensorDataset(train_X, train_Y), batch_size=config.batch_size)    # DataLoader可自动生成可训练的batch数据

    valid_X, valid_Y = torch.from_numpy(valid_X).float(), torch.from_numpy(valid_Y).float()
    valid_loader = DataLoader(TensorDataset(valid_X, valid_Y), batch_size=config.batch_size)

    device = torch.device("cuda:0" if config.use_cuda and torch.cuda.is_available() else "cpu") # CPU训练还是GPU
    if config.choose_model == "基础LSTM+双层全连接":# 如果是GPU训练， .to(device) 会把模型/数据复制到GPU显存中
        model = LSTM_Base_Net(config).to(device)
    elif config.choose_model == "改进LSTM+双层全连接":
        model = improved_LSTM_solo_Net(config).to(device)
    elif config.choose_model == "多任务LSTM":
        model = improved_LSTM_multi_Net(config).to(device)


    if config.add_train:                # 如果是增量训练，会先加载原模型参数
        model.load_state_dict(torch.load(config.model_save_path + config.model_name))
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = torch.nn.MSELoss()      # 这两句是定义优化器和loss

    valid_loss_min = float("inf")
    bad_epoch = 0
    global_step = 0
    for epoch in range(config.epoch):
        logger.info("Epoch {}/{}".format(epoch, config.epoch))
        model.train()                   # pytorch中，训练时要转换成训练模式
        train_loss_array = []
        hidden_train = None
        # pred_train_Y_list = []
        for i, _data in enumerate(train_loader):
            _train_X, _train_Y = _data[0].to(device),_data[1].to(device)
            optimizer.zero_grad()               # 训练前要将梯度信息置 0
            pred_Y, hidden_train = model(_train_X, hidden_train)    # 这里走的就是前向计算forward函数

            if not config.do_continue_train:
                hidden_train = None             # 如果非连续训练，把hidden重置即可
            else:
                h_0, c_0 = hidden_train
                h_0.detach_(), c_0.detach_()    # 去掉梯度信息
                hidden_train = (h_0, c_0)
            loss = criterion(pred_Y, _train_Y)  # 计算loss
            # pred_train_Y_list.append(pred_Y)
            loss.backward()                     # 将loss反向传播
            optimizer.step()                    # 用优化器更新参数
            train_loss_array.append(loss.item())
            global_step += 1
            if config.do_train_visualized and global_step % 100 == 0:   # 每一百步显示一次
                vis.line(X=np.array([global_step]), Y=np.array([loss.item()]), win='Train_Loss',
                         update='append' if global_step > 0 else None, name='Train', opts=dict(showlegend=True))

        # 以下为早停机制，当模型训练连续config.patience个epoch都没有使验证集预测效果提升时，就停止，防止过拟合
        model.eval()                    # pytorch中，预测时要转换成预测模式
        valid_loss_array = []
        hidden_valid = None
        # pred_vaild_Y_list = []
        for _valid_X, _valid_Y in valid_loader:
            _valid_X, _valid_Y = _valid_X.to(device), _valid_Y.to(device)
            pred_Y, hidden_valid = model(_valid_X, hidden_valid)
            # pred_vaild_Y_list.append(pred_Y)
            if not config.do_continue_train: hidden_valid = None
            loss = criterion(pred_Y, _valid_Y)  # 验证过程只有前向计算，无反向传播过程
            valid_loss_array.append(loss.item())

        train_loss_cur = np.mean(train_loss_array)
        valid_loss_cur = np.mean(valid_loss_array)
        # predict_y = np.asarray(pred_train_Y_list)

        # # train_four_error = four_errors_of_model_result(label_y=np.asarray(train_Y).reshape(-1, 1),
        #                                                predict_y=np.asarray(pred_train_Y_list).reshape(-1, 1))
        # # valid_four_error = four_errors_of_model_result(label_y=np.asarray(valid_Y).reshape(-1, 1),
        #                                                predict_y=np.asarray(pred_vaild_Y_list).reshape(-1, 1))

        # logger.info("The train loss is {:.6f}. ".format(train_loss_cur) +
        #       "The valid loss is {:.6f}.".format(valid_loss_cur) + "\n" +
        #             "The train four losses are" + train_four_error + "\n" +
        #             "The vaild four losses are" + valid_four_error)
        logger.info("The train loss is {:.6f}. ".format(train_loss_cur) +
                    "The valid loss is {:.6f}.".format(valid_loss_cur))
        if config.do_train_visualized:      # 第一个train_loss_cur太大，导致没有显示在visdom中
            vis.line(X=np.array([epoch]), Y=np.array([train_loss_cur]), win='Epoch_Loss',
                     update='append' if epoch > 0 else None, name='Train', opts=dict(showlegend=True))
            vis.line(X=np.array([epoch]), Y=np.array([valid_loss_cur]), win='Epoch_Loss',
                     update='append' if epoch > 0 else None, name='Eval', opts=dict(showlegend=True))

        if valid_loss_cur < valid_loss_min:
            valid_loss_min = valid_loss_cur
            bad_epoch = 0
            torch.save(model.state_dict(), config.model_save_path + config.model_name)  # 模型保存
        else:
            bad_epoch += 1
            if bad_epoch >= config.patience:    # 如果验证集指标连续patience个epoch没有提升，就停掉训练
                logger.info(" The training stops early in epoch {}".format(epoch))
                # break
                pass


def predict(config, logger=None, test_X=None):
    # 获取测试数据
    test_X = torch.from_numpy(test_X).float()
    test_set = TensorDataset(test_X)
    test_loader = DataLoader(test_set, batch_size=1)

    # 加载模型
    device = torch.device("cuda:0" if config.use_cuda and torch.cuda.is_available() else "cpu")

    if config.choose_model == "基础LSTM+双层全连接":# 如果是GPU训练， .to(device) 会把模型/数据复制到GPU显存中
        model = LSTM_Base_Net(config).to(device)
    elif config.choose_model == "改进LSTM+双层全连接":
        model = improved_LSTM_solo_Net(config).to(device)
    elif config.choose_model == "多任务LSTM":
        model = improved_LSTM_multi_Net(config).to(device)

    # model = Net(config).to(device)
    model.load_state_dict(torch.load(config.model_save_path + config.model_name))   # 加载模型参数

    # 先定义一个tensor保存预测结果
    result = torch.Tensor().to(device)

    # 预测过程
    model.eval()
    hidden_predict = None
    start_time = time()
    for _data in test_loader:
        data_X = _data[0].to(device)
        pred_X, hidden_predict = model(data_X, hidden_predict)
        # if not config.do_continue_train: hidden_predict = None    # 实验发现无论是否是连续训练模式，把上一个time_step的hidden传入下一个效果都更好
        cur_pred = torch.squeeze(pred_X, dim=0)
        if config.choose_model == "多任务LSTM":
            cur_pred = cur_pred.permute(1, 0)
            result = torch.cat((result, cur_pred), dim=0)
        else:
            result = torch.cat((result, cur_pred), dim=0)

    end_time = time()
    test_time = (end_time - start_time) / test_X.shape[0]
    logger.info(" 测试时间为 {}".format(test_time))


    return result.detach().cpu().numpy()    # 先去梯度信息，如果在gpu要转到cpu，最后要返回numpy数据

