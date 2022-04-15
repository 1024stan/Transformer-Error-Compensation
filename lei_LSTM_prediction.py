#!/usr/bin/env python 
# -*- coding:utf-8 -*-
from LSTM_regression import *
from tools import *
from time import *
import time

class Config:
    # 数据参数
    file_name = './data/huganqi/all_change.xlsx'
    sheet_name = '所有'
    feature_columns = list(range(2, 9))     # 要作为feature的列，按原数据从0开始计算，也可以用list 如 [2,4,6,8] 设置
    label_columns = [4, 5]                  # 要预测的列，按原数据从0开始计算, 如同时预测第四，五列 最低价和最高价
    # label_in_feature_index = [feature_columns.index(i) for i in label_columns]  # 这样写不行
    label_in_feature_index = (lambda x,y: [x.index(i) for i in y])(feature_columns, label_columns)  # 因为feature不一定从0开始

    predict_day = 1             # 预测未来几天

    # 网络参数
    # input_size = len(feature_columns)
    # output_size = len(label_columns)
    input_size = 7
    output_size = 1
    use_bidirectional = True    # 使用双向
    choose_model = "多任务LSTM"   # 基础LSTM+双层全连接、改进LSTM+双层全连接、多任务LSTM

    hidden_size = 128           # LSTM的隐藏层大小，也是输出大小
    lstm_layers = 2             # LSTM的堆叠层数
    dropout_rate = 0.1          # dropout概率
    time_step = 20              # 这个参数很重要，是设置用前多少天的数据来预测，也是LSTM的time step数，请保证训练数据量大于它
    history_times = 10   ## 历史时间长度
    # 卷积网络参数
    con_output_size = 7
    kernel_size = 4
    num_convs_layers = 10

    # 多任务最后层参数
    num_fc_layers = 3

    # 训练参数
    do_train = True
    do_predict = True
    add_train = False           # 是否载入已有模型参数进行增量训练
    shuffle_train_data = True   # 是否对训练数据做shuffle
    use_cuda = True            # 是否使用GPU训练

    train_data_rate = 0.9      # 训练数据占总体数据比例，测试数据就是 1-train_data_rate
    valid_data_rate = 0.1      # 验证数据占训练数据比例，验证集在训练过程使用，为了做模型和参数选择

    batch_size = 64
    learning_rate = 0.001
    epoch = 20                  # 整个训练集被训练多少遍，不考虑早停的前提下
    patience = 5                # 训练多少epoch，验证集没提升就停掉
    random_seed = 42            # 随机种子，保证可复现

    do_continue_train = False    # 每次训练把上一次的final_state作为下一次的init_state，仅用于RNN类型模型，目前仅支持pytorch
    continue_flag = ""           # 但实际效果不佳，可能原因：仅能以 batch_size = 1 训练
    if do_continue_train:
        shuffle_train_data = False
        batch_size = 1
        continue_flag = "continue_"

    # 训练模式
    debug_mode = False  # 调试模式下，是为了跑通代码，追求快
    debug_num = 500  # 仅用debug_num条数据来调试

    # 框架参数
    used_frame = "pytorch"  # 选择的深度学习框架，不同的框架模型保存后缀不一样
    model_postfix = {"pytorch": ".pth", "keras": ".h5", "tensorflow": ".ckpt"}
    model_name = "model_" + continue_flag + used_frame + model_postfix[used_frame]

    # 路径参数
    train_data_path = "./data/huganqi/all_change.xlsx"
    model_save_path = "./checkpoint/" + used_frame + "/"
    figure_save_path = "./figure/"
    log_save_path = "./log/"
    do_log_print_to_screen = True
    do_log_save_to_file = True                  # 是否将config和训练过程记录到log
    do_figure_save = False
    do_train_visualized = False          # 训练loss可视化，pytorch用visdom，tf用tensorboardX，实际上可以通用, keras没有
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)    # makedirs 递归创建目录
    if not os.path.exists(figure_save_path):
        os.mkdir(figure_save_path)
    if do_train and (do_log_save_to_file or do_train_visualized):
        cur_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        log_save_path = log_save_path + cur_time + '_' + used_frame + "/"
        os.makedirs(log_save_path)



def main(config, choosen):
    logger = load_logger(config)
    try:
        np.random.seed(config.random_seed)  # 设置随机种子，保证可复现
        if config.choose_model == "多任务LSTM":
            data_gainer = two_output_LSTM_Data(config)
        else:
            data_gainer = LSTM_Data(config)

        if config.do_train:
            train_X, valid_X, train_Y, valid_Y = data_gainer.get_train_and_valid_data(flag=choosen)
            train(config, logger, [train_X, train_Y, valid_X, valid_Y])

        if config.do_predict:
            test_X, test_Y = data_gainer.get_all_data(flag=choosen)
            pred_result = predict(config, logger, test_X)       # 这里输出的是未还原的归一化预测数据
            if config.choose_model == "多任务LSTM":
                errors = four_errors_of_model_result(label_y=test_Y[:, 0], predict_y=pred_result[:,0])
                print('比差结果误差：\n')
                print(errors)

                errors = four_errors_of_model_result(label_y=test_Y[:, 0], predict_y=pred_result[:, 1])
                print('角差结果误差：\n')
                print(errors)
            else:
                errors = four_errors_of_model_result(label_y=test_Y.reshape(test_Y.shape[0], test_Y.shape[1]),
                                                     predict_y=pred_result)
                print(errors)
            # print(pred_result)
            # draw(config, data_gainer, logger, pred_result)
    except Exception:
        logger.error("Run Error", exc_info=True)



if __name__=="__main__":
    import argparse
    # argparse方便于命令行下输入参数，可以根据需要增加更多
    parser = argparse.ArgumentParser()
    # parser.add_argument("-t", "--do_train", default=False, type=bool, help="whether to train")
    # parser.add_argument("-p", "--do_predict", default=True, type=bool, help="whether to train")
    # parser.add_argument("-b", "--batch_size", default=64, type=int, help="batch size")
    # parser.add_argument("-e", "--epoch", default=20, type=int, help="epochs num")
    args = parser.parse_args()

    con = Config()
    for key in dir(args):               # dir(args) 函数获得args所有的属性
        if not key.startswith("_"):     # 去掉 args 自带属性，比如__name__等
            setattr(con, key, getattr(args, key))   # 将属性值赋给Config

    main(con, choosen="jiaocha")




