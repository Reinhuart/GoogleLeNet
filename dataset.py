import os
import random
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def create_dataset(path, batch_size):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    Datasets = datasets.ImageFolder(path, transform)
    dataloader = DataLoader(Datasets, batch_size, shuffle=True)
    return dataloader


def one_hot(ytrain, path):
    dir_path = path + '/'
    list_classic = os.listdir(dir_path)
    y = torch.zeros((len(ytrain), len(list_classic)))
    for i in range(len(ytrain)):
        y[i, ytrain[i]] = 1
        pass
    return y


def show_img(path):
    animal_classic = os.listdir(path + '/')
    choice = []
    for i in range(3):
        num = random.randint(0, len(animal_classic) - 1)
        choice.append(animal_classic[num])
        pass

    plt.figure(dpi=250)
    pos = 1
    for j in choice:
        dir_path = path + '/' + j
        img_list = os.listdir(dir_path)
        for k in range(3):
            num = random.randint(0, len(img_list) - 1)
            img = plt.imread(dir_path + '/' + img_list[num])
            plt.subplot(3, 3, pos)
            plt.imshow(img)
            plt.title(j)
            plt.axis('off')
            pos += 1
            pass
        pass
    plt.show()

    return None


if __name__ == '__main__':
    path = r'G:\Python代码\Python代码\pycharm\机器学习\learn AI with LiMu\动物分类\animals'
    dataloader = create_dataset(path, 50)
    show_img(path)
