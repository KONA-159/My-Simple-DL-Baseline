import os

import pandas
import yaml
import torch.nn as nn
import torch.optim as optim

import utils
import warnings
class HyperParameters:
    batch_size: int = None
    epoch: int = None
    learning_rate: float = None
    # 类
    model: nn.Module = None
    # 类
    loss_function = None
    # 类
    optimizer=None
    weight_decay=None

    def __init__(self) -> None:
        super().__init__()
        warnings.warn("已弃用", DeprecationWarning)

class Config:
    __instance = None
    category_num=None
    is_rgb = None
    device = None
    image_root_directory = None
    training_csv = None
    testing_csv = None
    k_fold=None
    def __init__(self):
        warnings.warn("已弃用", DeprecationWarning)
        self.hyper_parameters = HyperParameters()
        with open('config.yaml', 'r', encoding='utf-8') as f:
            config_yaml = yaml.load(f.read(), Loader=yaml.FullLoader)
        self.hyper_parameters.batch_size = config_yaml['hyper_parameter']['batch_size']
        self.hyper_parameters.epoch = config_yaml['hyper_parameter']['epoch']
        self.hyper_parameters.learning_rate = config_yaml['hyper_parameter']['learning_rate']
        import models
        self.hyper_parameters.model = getattr(models, config_yaml['hyper_parameter']['model'])
        self.hyper_parameters.optimizer=getattr(optim,config_yaml['hyper_parameter']['optimizer'])
        self.hyper_parameters.loss_function=getattr(nn,config_yaml['hyper_parameter']['loss_function'])
        self.hyper_parameters.weight_decay=config_yaml['hyper_parameter']['weight_decay']
        self.k_fold=config_yaml['k_fold']
        self.category_num=config_yaml['category_num']
        self.is_rgb = config_yaml['is_rgb']
        self.device = utils.try_gpu(config_yaml['device'])
        self.image_root_directory = os.path.join(config_yaml['dataset_root'])
        self.training_csv = os.path.join(config_yaml['dataset_root'], config_yaml['training_set'])
        self.testing_csv = os.path.join(config_yaml['dataset_root'], config_yaml['training_set'])

    @staticmethod
    def get_instance():
        warnings.warn("已弃用", DeprecationWarning)
        if Config.__instance is None:
            Config.__instance = Config()
        return Config.__instance
