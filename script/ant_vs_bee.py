from pathlib import Path

import torch
from sklearn.metrics import accuracy_score
from torch import optim, nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from script.dataset import myDataset
from script.simple_cnn import SimpleCNN


def run_train_epoch(model, train_loader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    running_acc = 0.0

    for images, labels in train_loader:
        # 勾配初期化
        optimizer.zero_grad()

        # 順伝播計算
        outputs = model(images)

        # 損失の計算
        loss = criterion(outputs, labels)

        # Backward
        loss.backward()

        # 重みの更新
        optimizer.step()

        _, labels_pred = torch.max(outputs, dim=1)
        running_loss += loss.item() * images.size(0)
        running_acc += accuracy_score(labels, labels_pred) * images.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = running_acc / len(train_loader.dataset)

    print(f'Epoch: {epoch} Train Loss: {epoch_loss:.4f} Train Acc: {epoch_acc:.4f}')


def run_valid_epoch(model, valid_loader, criterion, epoch):
    model.eval()
    running_loss = 0.0
    running_acc = 0.0

    with torch.no_grad():
        for images, labels in valid_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)

            _, labels_pred = torch.max(outputs, dim=1)
            running_loss += loss.item() * images.size(0)
            running_acc += accuracy_score(labels, labels_pred) * images.size(0)

    epoch_loss = running_loss / len(valid_loader.dataset)
    epoch_acc = running_acc / len(valid_loader.dataset)
    print(f'Epoch: {epoch} Valid Loss: {epoch_loss:.4f} Valid Acc: {epoch_acc:.4f}')


def acc_score(net, valid_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in valid_loader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy of the network on the 10000 test images: {100 * correct / total:.2f} %')


def class_acc(net, valid_loader):
    classes = (0, 1)
    class_correct = [0., 0.]
    class_total = [0., 0.]
    with torch.no_grad():
        for data in valid_loader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    for i in range(2):
        print(f'Accuracy of {classes[i]} : {100 * class_correct[i] / class_total[i]:.4f} %')


def main():
    torch.manual_seed(0)
    train_transform = transforms.Compose([
        transforms.Resize(size=(200, 200)),
        transforms.ToTensor()
    ])

    # データローダーの準備
    BATCH_SIZE = 64
    train_path = Path('../hymenoptera_data/train')
    train_dataset = myDataset(train_path, train_transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_path = Path('../hymenoptera_data/val')
    valid_dataset = myDataset(valid_path, train_transform)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # モデル構築
    net = SimpleCNN()
    optimizer = optim.Adam(net.parameters())
    criterion = nn.CrossEntropyLoss()

    # 3. 学習
    NUM_EPOCHS = 5
    for epoch in tqdm(range(1, NUM_EPOCHS + 1)):
        run_train_epoch(net, train_loader, criterion, optimizer, epoch)
        run_valid_epoch(net, valid_loader, criterion, epoch)

    # 検証データの正答数
    # 全体の正答率
    acc_score(net, valid_loader)
    # クラス毎の正答率
    class_acc(net, valid_loader)


if __name__ == '__main__':
    main()
