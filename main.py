# This is a sample Python script.
import os
import time
from datetime import datetime

import pandas
import torch
from torch import nn
from sklearn.model_selection import KFold
import metrics
import utils
from configv1 import Config
from dataset import LeafDataset
import torchvision.transforms as transforms
import torch.utils.data as data
from torch.optim import lr_scheduler
from scheduler import get_scheduler
import torch.multiprocessing as mp


def load_training_set():
    data_frame = pandas.read_csv(Config.training_csv)  # 读训练集df
    preprocess = Config.data_transforms['train']  # 预处理
    training_set = LeafDataset(data_frame, transform=preprocess,
                               is_rgb=Config.is_rgb,
                               has_label=True
                               )  # 封装成数据集
    return training_set


def load_testing_set():
    data_frame = pandas.read_csv(Config.testing_csv)
    preprocess = Config.data_transforms['valid']
    testing_set = LeafDataset(data_frame, transform=preprocess,
                              is_rgb=Config.is_rgb,
                              has_label=False
                              )
    return testing_set


def set_optimizer(model, is_param_grouped):
    if is_param_grouped:#前后使用不同的学习率
        params_feature_extract = [param for name, param in model.named_parameters() if not any(
            name.__contains__(substring) for substring in ['fc.weight', 'fc.bias'])]  # any:如果有任何一个true就停止迭代直接返回
        params_fully_connnected = [param for name, param in model.named_parameters() if
                                   any(name.__contains__(substring) for substring in ['fc.weight', 'fc.bias'])]
        optimizer: torch.optim.Optimizer = (  # 设置优化器
            Config
            .hyper_parameters
            .optimizer([{"params":params_feature_extract},
                        {"params":params_fully_connnected,"lr":Config.hyper_parameters.learning_rate*Config.hyper_parameters.fine_tune_boost}],
                       lr=Config.hyper_parameters.learning_rate, weight_decay=Config.hyper_parameters.weight_decay))
        return optimizer
    else:
        params_weight = (p for name, p in model.named_parameters() if 'weight' in name)
        params_bias = (param for name, param in model.named_parameters() if 'bias' in name)
        optimizer: torch.optim.Optimizer = (  # 设置优化器
            Config
            .hyper_parameters
            .optimizer([{"params": params_weight,  # 找到模型中所有名字带有weight的参数
                         'weight_decay': Config.hyper_parameters.weight_decay},  # 设置weight-decay
                        {'params': params_bias}],
                       lr=Config.hyper_parameters.learning_rate))
        return optimizer


def train_with_test(training_iter, testing_iter=None, fold=0):
    # 设置模型
    model: nn.Module = Config.hyper_parameters.model(categories=Config.category_num,
                                                     is_rgb=Config.is_rgb,
                                                     )  # 不要在config构造方法中实例化，在config构造时实例化会导致model在实例化时读取config中的属性导致类的构造方法循环依赖
    pretrained = Config.hyper_parameters.pretrained
    is_param_grouped = pretrained

    optimizer = set_optimizer(model, is_param_grouped)

    utils.model_init(model, pretrained=pretrained)  # 模型初始化
    model.to(Config.device)  # 移动至gpu
    scheduler = get_scheduler(optimizer)  # 设置学习率调整器
    loss_function = Config.hyper_parameters.loss_function()  # 设置损失函数
    train_metric = metrics.Metrics()  # 初始化评估器
    test_metric = metrics.Metrics()
    # 开始训练
    for epoch in range(Config.hyper_parameters.epoch):  # 每个epoch
        start_time_epoch = time.time()  # epoch开始时间
        model.train()  # train()要尽可能接近底层，不然如果循环中有推理，可能会忘记设置它
        train_metric.reset()  # 评估器归零
        for sample, label in training_iter:  # 每批数据
            sample = sample.to(Config.device)  # 移动到gpu
            label = label.to(Config.device)  # 差点漏了
            optimizer.zero_grad()  # 清空参数的梯度
            train_predict = model(sample)  # 推理，一个batch是一起算的，每个batch更新一次weight，所以做多gpu的时候可以异步地分开，用batch是为了快速迭代数据的同时保持梯度下降的稳定
            loss: torch.Tensor = loss_function(train_predict, label)  # 计算batch平均损失
            loss.backward()  # 求梯度，torch官方的cross-entropy是整合了softmax分类+计算argmax+cross-entropy的，所以无须更改
            optimizer.step()  # 更新参数

            correct_num = utils.correct_num(train_predict, label)  # 正确预测数量
            train_metric.add_in_epoch(loss=loss.item(), sample_num=label.numel(), correct=correct_num)  # 记录loss

        # 评估部分
        if testing_iter is not None:  # 有测试集
            test_metric.reset()  # 评估器归零
            model.eval()  # 进入评估模式忽略BN和Dropout
            with torch.no_grad():
                for sample, label in testing_iter:
                    sample = sample.to(Config.device)  # 移动到gpu
                    label = label.to(Config.device)  # 差点漏了
                    test_predict = model(sample)  # 推理
                    correct_num = utils.correct_num(test_predict, label)  # 判断正确数量
                    test_loss = loss_function(test_predict, label)  # 计算loss
                    test_metric.add_in_epoch(sample_num=label.numel(), correct=correct_num,
                                             loss=test_loss.item())  # 评估器更新

            if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):  # 判断学习率调度器类型
                scheduler.step(test_metric.average_loss_epoch())  # 更新学习率
            else:
                scheduler.step()
            end_time_epoch = time.time()  # epoch结束时间
            elapse = end_time_epoch - start_time_epoch
            utils.SUMMARY_WRITER.add_scalars(f'Fold{fold}/Loss', {"train loss": train_metric.average_loss_epoch(),
                                                                  "test loss": test_metric.average_loss_epoch(), },
                                             epoch)  # tensorboard记录loss
            utils.SUMMARY_WRITER.add_scalars(f'Fold{fold}/Accuracy', {"train accuracy": train_metric.accuracy_epoch(),
                                                                      "test accuracy": test_metric.accuracy_epoch()},
                                             epoch)  # tensorboard记录acc
            utils.LOGGER.info(
                f'Epoch {epoch}/{Config.hyper_parameters.epoch - 1}:train_loss = {train_metric.average_loss_epoch():.4f},test_loss = {test_metric.average_loss_epoch():.4f},train_accuracy = {train_metric.accuracy_epoch():.4f},test_accuracy = {test_metric.accuracy_epoch():.4f},time: {elapse:.0f}s')  # 日志记录
    if testing_iter is not None:
        if test_metric.accuracy_epoch() > Config.qualified_accuracy:  # TODO fold最终准确率>标准，则存储模型，没什么用，以后优化掉
            model_save_path = os.path.join(Config.model_save_root,
                                           Config.hyper_parameters.model_name + f'_fold:{fold}_acc:{test_metric.accuracy_epoch():.4f}' + '.pth')
            torch.save(model.state_dict(), model_save_path)  # 存储模型
            utils.LOGGER.info(f'Save Qualified Score: {test_metric.accuracy_epoch():.4f} Model to {model_save_path}')
        return test_metric.accuracy_epoch(), train_metric.accuracy_epoch()  # 返回该fold最终准确率
    else:
        model_save_path = os.path.join(Config.model_save_root,
                                       Config.hyper_parameters.model_name + '.pth')
        torch.save(model.state_dict(), model_save_path)  # 直接存
        utils.LOGGER.info(f'Save Model to {model_save_path}')
    return model


def inference():
    test_data_frame = pandas.read_csv(Config.testing_csv)  # 读csv
    testing_set = load_testing_set()  # 读测试集
    testing_iter = data.DataLoader(dataset=testing_set, batch_size=Config.hyper_parameters.batch_size,
                                   num_workers=Config.num_workers)
    model: nn.Module = Config.hyper_parameters.model(categories=Config.category_num,
                                                     is_rgb=Config.is_rgb)  # 设置模型
    model.to(Config.device)  # 模型移动到gpu
    load_path = Config.model_load_path
    model.load_state_dict(torch.load(load_path))  # 加载模型参数
    model.eval()  # 评估模式
    test_data_frame.insert(test_data_frame.shape[1], 'label', value='NaN')  # csv加一列label
    test_data_frame['label'].astype('str')  # 要先改变df的数据类型否则str会丢失
    for image_relative_location_batch, sample_batch in testing_iter:  # 读一个batch
        with torch.no_grad():
            predict = model(sample_batch)  # 推理
            category_num_list = utils.softmax_classifier(predict).tolist()  # 分类并转为list，tensor转list之后才能进入dict
            for image_relative_location, category_num in zip(image_relative_location_batch,
                                                             category_num_list):  # 使用zip同时迭代两个list，如果两个列表长度不同则会以最短的来迭代
                category_str = utils.NUM_TO_CATEGORY[category_num]  # num映射为str
                test_data_frame.loc[
                    test_data_frame['image'] == image_relative_location, 'label'] = category_str  # dict不能直接传入list，迭代来做
    test_data_frame.head()  # csv加个头，满足kaggle上传需要
    test_data_frame.to_csv(Config.predict_result, index=False)  # 如果没有的话得自己创建


def k_fold_validate(data_frame, k):
    preprocess_train = Config.data_transforms['train']
    preprocess_test = Config.data_transforms['valid']
    total_accuracy = 0.0
    kfold = KFold(n_splits=k, shuffle=True, random_state=Config.seed)  # 使用sklearn的K折分割器
    batch_size = Config.hyper_parameters.batch_size
    fold = 0
    for training_fold_indices, testing_fold_indices in kfold.split(data_frame):  # 返回每一折的index用于划分
        utils.LOGGER.info(f'Fold {fold}/{k - 1} :')

        training_frame = data_frame.iloc[training_fold_indices]  # 在csv里读，然后构造数据集，这样可以分开预处理
        testing_frame = data_frame.iloc[testing_fold_indices]

        training_fold = LeafDataset(training_frame, preprocess_train, is_rgb=Config.is_rgb, has_label=True)
        testing_fold = LeafDataset(testing_frame, preprocess_test, is_rgb=Config.is_rgb, has_label=True)
        training_iter = data.DataLoader(training_fold, batch_size=batch_size, shuffle=True,
                                        num_workers=Config.num_workers)
        testing_iter = data.DataLoader(testing_fold, batch_size=batch_size, shuffle=True,
                                       num_workers=Config.num_workers)

        test_accuracy, _ = train_with_test(training_iter, testing_iter, fold)  # 计算总的准确率

        fold += 1
        total_accuracy += test_accuracy
    average_accuracy = total_accuracy / k
    utils.LOGGER.info(f'Total accuracy:{average_accuracy:.4f}')
    return average_accuracy


if __name__ == '__main__':
    mp.set_start_method('spawn')#多进程getitem开关
    if Config.mode == 'validate':
        data_frame = pandas.read_csv(Config.training_csv)
        k_fold_validate(data_frame, Config.k_fold)
    elif Config.mode == 'train':
        training_set = load_training_set()
        training_iter = data.DataLoader(training_set, batch_size=Config.hyper_parameters.batch_size, shuffle=True,
                                        num_workers=Config.num_workers)
        train_with_test(training_iter)
    elif Config.mode == 'inference':
        inference()
