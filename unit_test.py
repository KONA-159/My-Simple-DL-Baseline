import os
import unittest

import pandas
import torch
import torchvision.transforms
from torch import optim
from torchvision.transforms import transforms

import main
import models
import utils
from configv1 import Config
from dataset import LeafDataset
from datetime import datetime


class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)  # add assertion here

    def test_correct_num(self):
        X=torch.tensor([[0.,1.,2.],[5.,4.,3.]])
        Y=torch.tensor([2.,0.])
        print(utils.correct_num(X,Y))

    def test_categories(self):
        preprocess = transforms.Compose([transforms.ToTensor()])
        training_set = LeafDataset(transform=preprocess,
                                   is_rgb=Config.is_rgb,
                                   has_label=True
                                   )
        print(training_set.get_category_encoder())
        print(training_set.__getitem__(6))

    def test_param(self):
        model=models.PreTrainedResNet18(176,True)
        # params_feature_extract=[name for name,param in model.named_parameters() if any(substring not in name for substring in ['fc.weight','fc.bias'])]
        # print(params_feature_extract)
        # model.apply(print)
        # print('model',model)
        # print('model.model[0]',model.model[0]#返回Sequential的第n个
        # print('model.model',model.model)#按名字返回子模块
        # print('model._modules',model.named_modules())#返回子模块列表迭代对象
        # for i in model.children():#直接子模块
        #     print(i)
        # print('----------------------------------------')
        # for i in model.modules():#所有模块
        #     print(i)
        # print('----------------------------------------')
        for name,module in model.named_modules():#模块名+模块
            print(name,type(module))
        # print('----------------------------------------')
        # print('model.parameters()',model.parameters())#获得无名所有参数迭代对象
        # print('model.named_parameters()',model.named_parameters())#获得有名所有参数迭代对象
        # for i in model.named_parameters():
        #     print(i)
        # print('model.state_dict()',model.state_dict())#返回有序字典
        # optim.Adam()

    def test_datetime(self):
        print(datetime.now().strftime('%Y-%m-%d-%H:%M:%S'))

    def test_logger(self):
        utils.LOGGER.info('log test')
        print(__name__)

    def test_tensorboard(self):
        import numpy as np
        from torch.utils.tensorboard import SummaryWriter
        #
        writer = SummaryWriter(comment='test_tensorboard')

        for x in range(100):
            writer.add_scalar('y=2x', x * 2, x)
            writer.add_scalar('y=pow(2, x)', 2 ** x, x)

            writer.add_scalars('data/scalar_group', {"xsinx": x * np.sin(x),
                                                     "xcosx": x * np.cos(x),
                                                     "arctanx": np.arctan(x)}, x)
        writer.close()

    def test_save_model_path(self):
        model_path =os.path.join(Config.model_save_root, Config.hyper_parameters.model_name + f'_fold:{4}_acc:{213.2132:.4f}' + '.pth')
        print(model_path)

    def test_mapping(self):
        print(utils.get_mappings())

    def test_timm_model(self):
        model=models.PreTrainedResNet50(categories=Config.category_num,is_rgb=Config.is_rgb)
        print(model)#timm最后全连接层也是fc

    def test_wandb(self):
        import wandb
        import random

        # start a new wandb run to track this script
        wandb.init(
            # set the wandb project where this run will be logged
            project="my-awesome-project",

            # track hyperparameters and run metadata
            config={
                "learning_rate": 0.02,
                "architecture": "CNN",
                "dataset": "CIFAR-100",
                "epochs": 10,
            }
        )

        # simulate training
        epochs = 10
        offset = random.random() / 5
        for epoch in range(2, epochs):
            acc = 1 - 2 ** -epoch - random.random() / epoch - offset
            loss = 2 ** -epoch + random.random() / epoch + offset

            # log metrics to wandb
            wandb.log({"acc": acc, "loss": loss})

        # [optional] finish the wandb run, necessary in notebooks
        wandb.finish()

if __name__ == '__main__':
    unittest.main()
