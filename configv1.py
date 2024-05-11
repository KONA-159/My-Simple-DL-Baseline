import os.path
from datetime import datetime

from torch import nn
from torch import optim
from torchvision import transforms

import utils
import timm


class Config:
    mode = 'validate'  # train,validate,inference
    # 路径
    dataset_root = './dataset'  # 数据集根目录
    result_root = './result'  # 结果根目录
    training_set = 'train.csv'  # 训练集名称
    testing_set = 'test.csv'  # 测试集名称
    images = 'images'  # 图像目录名称
    num_workers = 0 #用于多进程getitem，由于是不同进程，使用不同时间戳会导致日志计入不同文件
    if num_workers != 0:
        begin_time = datetime.now().strftime("%Y-%m-%d")  # 当前时间用于记录结果
        training_comment='multWorkersTest'
        predict_result = os.path.join(result_root, f'predicts/{begin_time}:{training_comment}.csv')  # 推理结果
        log_file = os.path.join(result_root, f'output_log/{begin_time}:{training_comment}')  # 日志文件
        summary_dir = os.path.join(result_root, f'summary/{begin_time}:{training_comment}')  # tensorboard可视化日志文件
    else:
        begin_time = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")  # 当前时间用于记录结果
        predict_result = os.path.join(result_root, f'predicts/{begin_time}.csv')  # 推理结果
        log_file = os.path.join(result_root, f'output_log/{begin_time}')  # 日志文件
        summary_dir = os.path.join(result_root, f'summary/{begin_time}')  # tensorboard可视化日志文件
    summary_comment = ''  # tensorboard标签
    model_save_root = './saved_models'  # 模型存储目录
    model_load_path = os.path.join(model_save_root, 'ResNet18.pth')  # 推理时读取的模型文件
    image_root_directory = os.path.join(dataset_root)  # 图像路径
    training_csv = os.path.join(dataset_root, training_set)  # 训练集路径
    testing_csv = os.path.join(dataset_root, testing_set)  # 测试集路径


    qualified_accuracy = 0.80  # 至少达到0.80才会保存
    device = utils.try_gpu('cuda:0')  # 使用的硬件
    is_rgb = True  # 图像rgb类型
    category_num = 176  # 分类数目
    k_fold = 5  # 交叉验证折数
    seed = 42  # 随机种子

    data_transforms = {  # 预处理
        'train':  # 训练预处理
            transforms.Compose([  # 预处理打包
                transforms.ToTensor(),  # 图像转换为tensor
                transforms.RandomHorizontalFlip(p=0.5),  # 水平翻转，p为概率
                transforms.RandomVerticalFlip(p=0.5),  # 上下翻转
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0),#上下浮动改变亮度、对比度、饱和度、色温
                transforms.RandomResizedCrop([299, 299],scale=(0.5,1.0)),  # 随机裁剪后resize，scale为保存的原图片的面积大小区间，ratio为高宽比
                transforms.Normalize(mean=[0.485, 0.456, 0.406],  # 使用ImageNet的mean和std正则化
                                     std=[0.229, 0.224, 0.225])
                # transforms.Normalize(mean=[0.5,0.5,0.5],  # 使用ImageNet的mean和std正则化
                #                      std=[0.5,0.5,0.5])
            ]),
        'valid':  # 测试集预处理
            transforms.Compose([
                transforms.ToTensor(),  # 同上
                transforms.Resize([299, 299]),  # 裁剪
                transforms.Normalize(mean=[0.485, 0.456, 0.406],  # 同上
                                     std=[0.229, 0.224, 0.225])
                # transforms.Normalize(mean=[0.5,0.5,0.5],
                #                      std=[0.5,0.5,0.5])
        ])
    }

    class hyper_parameters:
        batch_size = 256
        epoch = 60
        loss_function = getattr(nn, 'CrossEntropyLoss')
        optimizer = getattr(optim, 'Adam')
        import models
        model_name = 'PreTrainedResNet18'
        model = getattr(models, model_name)  # 自定义模型
        pretrained = model_name.__contains__('PreTrained')
        learning_rate = 8e-4
        fine_tune_boost=10#在微调时，线性层的lr的系数
        weight_decay = 1e-9  # weight-decay每次缩小的系数

        class learning_rate_schedule:
            scheduler = 'CosineAnnealingWarmRestarts'

            class ReduceLROnPlateau:  # 一段时间没降loss就降低学习率
                factor = 0.2  # 学习率变化系数
                patience = 4  # 容许的没有进步的次数
                eps = 1e-6  # 判断loss是否进步的标准

            class CosineAnnealingLR:  # 按照1/4个周期的余弦函数下降学习率
                T_max = 20  # 经过这么多个epoch之后进入下一个周期
                min_lr = 1e-6  # 最小学习率，最大学习率为初始学习率

            class CosineAnnealingWarmRestarts:  # 跟上面的一样，但是每次重启都把周期变长，整个曲线更平缓
                T_0 = 4  # 初始经过这么多epoch后进入下一个周期
                T_mult = 2  # 每次重启T_0乘以这个系数
                min_lr = 1e-6  # 同上
