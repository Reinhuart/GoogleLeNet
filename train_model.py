from GoogleLeNet import google_lenet
from dataset import create_dataset, one_hot
import torch
import torch.nn as nn


def accuary_num(y_hat, y):
    all = y.shape[0]
    num = 0
    for i in range(len(y)):
        if torch.argmax(y[i]) == torch.argmax(y_hat[i]):
            num += 1
        else:
            pass
        pass
    return round(num / all, 2)


def model_train(path, batch_size, lr, epoch):
    dataloader = create_dataset(path, batch_size)
    model = google_lenet()
    device = torch.device('cuda:0')
    model = model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    l = nn.CrossEntropyLoss(reduction='mean')
    history = []
    num = 0

    for i in range(epoch):
        for x, y in dataloader:
            x = x.to(device)
            y = one_hot(ytrain=y, path=path)
            y = y.to(device)

            # 运行模型，得到输出
            y_hat = model(x)
            accuary = accuary_num(y_hat, y)
            print(accuary)

            # 计算损失函数
            loss = l(y_hat, y)
            print(loss)
            history.append([float(loss), accuary])
            # 反向传播
            loss.backward()

            # 更新权重
            optimizer.step()
            # 清空梯度
            optimizer.zero_grad()

            # 判断准确度是否达标
            pass
        if accuary >= 0.98:
            break
        pass
    return history, model
