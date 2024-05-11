import pandas
import torch.cuda
from torch import nn
from torch.utils.tensorboard import SummaryWriter


def try_gpu(gpu_str):
    device_word = gpu_str.split(':')  # 把str拆分为cuda和标号
    gpu = int(device_word[1])  # 找到标号
    if torch.cuda.device_count() >= gpu + 1:  # 康康是否满足已有的gpu数量
        return torch.device(gpu_str)
    print('You are using CPU!')
    return torch.device('cpu')


def model_init(model:nn.Module,pretrained=False):
    # def module_init(module):
    #     if isinstance(module, nn.Linear):  # 线性层
    #         nn.init.xavier_normal_(module.weight)  # xavier初始化
    #         nn.init.constant_(module.bias, 0)  # bias设置成0
    #     elif isinstance(module, nn.Conv2d):  # 2维卷积
    #         nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
    #     elif isinstance(module, nn.BatchNorm2d):  # BN层
    #         nn.init.constant_(module.weight, 1)
    #         nn.init.constant_(module.bias, 0)
    #
    # model.apply(module_init)
    for name,module in model.named_modules():
        if isinstance(module, nn.Linear):  # 线性层
            nn.init.xavier_normal_(module.weight)  # xavier初始化
            nn.init.constant_(module.bias, 0)  # bias设置成0
        else:
            if not pretrained:
                if isinstance(module, nn.Conv2d):  # 2维卷积
                    nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')  # TODO 搞懂
                elif isinstance(module, nn.BatchNorm2d):  # BN层
                    nn.init.constant_(module.weight, 1)
                    nn.init.constant_(module.bias, 0)

def correct_num(predict_regression: torch.Tensor, label: torch.Tensor):
    with torch.no_grad():
        predict_num: torch.Tensor = softmax_classifier(predict_regression)  # 对回归结果进行softmax分类
        compare_tensor: torch.Tensor = predict_num.type(label.dtype) == label  # 判断batch内的分类结果和label是否相等
        correct_num = compare_tensor.sum().item()  # 把相等数加和
    return correct_num


def softmax_classifier(predict_regression: torch.Tensor):
    with torch.no_grad():
        predict_num = predict_regression.softmax(dim=1).argmax(dim=1)  # 看看哪个分类结果的回归结果最大，没有加softmax，因为比较省时间
    return predict_num


from configv1 import Config


def init_logger(log_file):
    from logging import getLogger, INFO, FileHandler, Formatter, StreamHandler

    logger = getLogger(__name__)
    logger.setLevel(INFO)  # 设置最低级别

    handler1 = StreamHandler()  # 标准输出
    handler1.setFormatter(Formatter("%(message)s"))  # 不加别的东西直接输出日志

    handler2 = FileHandler(filename=log_file)  # 日志持久化
    handler2.setFormatter(Formatter("%(message)s"))  # 同上
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


def get_mappings():  # 获取label字符串-数字的映射
    data_frame = pandas.read_csv(Config.training_csv)  # 读训练集df
    category_name_list = sorted(set(data_frame['label']))  # 排序去重
    category_to_num = dict(zip(category_name_list, range(len(category_name_list))))  # 把字符串和数字打包放进字典
    num_to_category = {value: key for key, value in category_to_num.items()}  # 反转字典
    num_class = len(category_name_list)  # 设置总数量
    Config.category_num = num_class
    return category_to_num, num_to_category


# 初始化
CATEGORY_TO_NUM, NUM_TO_CATEGORY = get_mappings()
LOGGER = init_logger(log_file=Config.log_file)
SUMMARY_WRITER = SummaryWriter(log_dir=Config.summary_dir, comment=Config.summary_comment)  # tensorboard日志初始化
