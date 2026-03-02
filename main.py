from train_model import model_train
from show_data import show
import torch

if __name__ == '__main__':
    path = r'F:\Python代码\Python代码\pycharm\机器学习\learn AI with LiMu\动物分类\animals'
    save_path = r'my_googlelenet_model.pth'
    lr = 0.1
    epoch = 5000
    batch_size = 100

    history, model = model_train(path, batch_size, lr, epoch)

    torch.save(model, save_path)

    show(history)