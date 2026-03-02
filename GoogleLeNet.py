#import netbios
import torch
import torch.nn as nn
from torchsummary import summary
import torch.nn.functional as F


class inception(nn.Module):
    def __init__(self, input_size, output_size, conv1_1):
        super(inception, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Conv2d(input_size, output_size[0], 1, 1, 0),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(output_size[0])
        )

        self.fc2 = nn.Sequential(
            nn.Conv2d(input_size, conv1_1[0], 1, 1, 0),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(conv1_1[0]),
            nn.Conv2d(conv1_1[0], output_size[1], 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(output_size[1])
        )

        self.fc3 = nn.Sequential(
            nn.Conv2d(input_size, conv1_1[1], 1, 1, 0),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(conv1_1[1]),
            nn.Conv2d(conv1_1[1], output_size[2], 5, 1, 2),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(output_size[2])
        )

        self.fc4 = nn.Sequential(
            nn.MaxPool2d(3, 1),
            nn.Conv2d(input_size, output_size[3], 1, 1, 1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(output_size[3])
        )

    def forward(self, x):
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        x3 = self.fc3(x)
        x4 = self.fc4(x)
        x_end = torch.cat([x1, x2, x3, x4], dim=1)
        return x_end


class google_lenet(nn.Module):
    def __init__(self):
        super(google_lenet, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(3, 2, 1),
            nn.BatchNorm2d(64)
        )
        self.fc2 = nn.Sequential(
            nn.Conv2d(64, 64, 1, 1, 0),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 192, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(3, 2, 1)
        )

        self.inception3a = inception(192, [64, 128, 32, 32], [96, 16])

        self.inception3b = inception(256, [128, 192, 96, 64], [128, 32])

        self.pool1 = nn.MaxPool2d(3, 2, 1)

        self.inception4 = nn.Sequential(
            inception(480, [192, 208, 48, 64], [96, 16]),
            inception(512, [160, 224, 64, 64], [112, 24]),
            inception(512, [128, 256, 64, 64], [128, 24]),
            inception(512, [112, 288, 64, 64], [114, 32]),
            inception(528, [256, 320, 128, 128], [160, 32])
        )

        self.pool2 = nn.MaxPool2d(3, 2, 1)

        self.inception5 = nn.Sequential(
            inception(832, [256, 320, 128, 128], [160, 32]),
            inception(832, [384, 384, 128, 128], [192, 48])
        )

        self.fc3 = nn.Sequential(
            nn.AvgPool2d(7, 1),
            nn.Dropout(0.4),
        )

        self.linear = nn.Linear(1024, 90)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.pool1(x)
        x = self.inception4(x)
        x = self.pool2(x)
        x = self.inception5(x)
        x = self.fc3(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        y_hat = F.softmax(x, dim=1)
        return y_hat


if __name__ == '__main__':
    device = torch.device('cuda:0')
    model = google_lenet()
    model = model.to(device)
    summary(model, (3, 224, 224))
