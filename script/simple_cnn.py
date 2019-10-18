import torch
from torch import nn


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 畳み込み層とプーリング層
        self.conv1 = nn.Conv2d(3, 6, 20)  # stride：カーネルの移動量
        self.conv2 = nn.Conv2d(6, 9, 20)
        self.conv3 = nn.Conv2d(9, 12, 20)

        self.relu = nn.ReLU(inplace=True)
        self.max_pool2d = nn.MaxPool2d(kernel_size=(2, 2))

        # 全結合層
        self.fc1 = nn.Linear(in_features=12*8*8, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=32)
        self.fc3 = nn.Linear(in_features=32, out_features=2)

    def forward(self, x):
        # 畳み込みとプーリング
        x = self.conv1(x)  # ->[6, 399, 399]
        x = self.relu(x)
        x = self.max_pool2d(x)  # ->[6, 199, 199]
        x = self.conv2(x)  # ->[9, 198, 198]
        x = self.relu(x)
        x = self.max_pool2d(x)  # ->[9, 99, 99]
        x = self.conv3(x)  # ->[9, 198, 198]
        x = self.relu(x)
        x = self.max_pool2d(x)  # ->[9, 99, 99]

        # 平滑化
        x = x.view(-1, 12 * 8 * 8)  # ->[88209]

        # 全結合
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)

        return x


def main():
    # size (N：データ数, C:チャンネル数, W:横，　H：縦)
    images = torch.ones(size=(32, 3, 200, 200))

    net = SimpleCNN()
    outputs = net(images)

    print(outputs.size())
    # assert outputs.size() == torch.Size([5, 10])


if __name__ == '__main__':
    main()
