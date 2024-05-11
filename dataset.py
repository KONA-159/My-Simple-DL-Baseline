import os.path
import pandas
import torch
import torchvision.transforms
import yaml
import torchvision.transforms as transforms
import torch.utils.data as data
import PIL.Image as Image
import utils
from configv1 import Config


class LeafDataset(data.Dataset):
    def __init__(self, data_frame, transform=None, is_rgb=None, has_label=True):
        self.__data_set = []  # 把dataframe读到内存
        self.__transform = transform  # 预处理
        self.__is_rgb = is_rgb
        self.__has_label = has_label
        self.__data_frame = data_frame
        if self.__has_label:  # 加载有标签的数据集
            for tuple in self.__data_frame.itertuples():  # 迭代df中的每一行，作为元组返回，0是index
                self.__data_set.append((tuple[1], utils.CATEGORY_TO_NUM[tuple[2]]))  # 把图像路径和label数字加入数据集
        else:  # 加载没有标签的数据集
            for tuple in self.__data_frame.itertuples():
                self.__data_set.append((tuple[1], None))

    def __getitem__(self, index):  # DataLoader在取数据返回iter时调用的方法
        image_relative_location, label = self.__data_set[index]  # 读到图像相对路径和label
        image_location = os.path.join(Config.image_root_directory, image_relative_location)  # 连接路径
        image = Image.open(image_location)  # 用PIL打开图像
        if self.__is_rgb:
            image = image.convert('RGB')  # 转换为RGB
        if self.__transform is not None:
            image = self.__transform(image)  # 预处理
            #getitem的时候，移动device的代码要放在外面，否则主进程移进显存了，子进程还要再移一次，显存就爆了，应该是子进程读进内存，主进程移到显存
        if self.__has_label:
            label = torch.tensor(label)  # label转换为tensor
            return image, label
        else:
            return image_relative_location, image

    def __len__(self):
        return len(self.__data_set)
