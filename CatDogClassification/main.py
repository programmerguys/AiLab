import matplotlib
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import torch
import torch.nn as nn
import cv2
import matplotlib.pyplot as plt
import torchvision
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms, models
from torch.optim.lr_scheduler import *
import copy
import random
import tqdm
from PIL import Image
import torch.nn.functional as F

labelList = ['cat', 'dog']
BATCH_SIZE = 20  # 每批次的大小
EPOCHS = 5  # 迭代次数
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 使用cpu或者gpu

cPath = os.getcwd()  # 获取当前路径
train_dir = cPath + '/data/train'  # 训练集路径
test_dir = cPath + '/data/test'  # 测试集路径
train_files = os.listdir(train_dir)  # 训练集文件名下的所有文件
test_files = os.listdir(test_dir)  # 测试集文件名下的所有文件


class CatDogDataset(Dataset):
    def __init__(self, file_list, dir, mode='train', transform=None):
        self.file_list = file_list
        self.dir = dir
        self.mode = mode
        self.transform = transform
        if self.mode == 'train':
            if 'dog' in self.file_list[0]:
                self.label = 1
            else:
                self.label = 0

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.dir, self.file_list[idx]))
        if self.transform:
            img = self.transform(img)
        if self.mode == 'train':
            img = img.numpy()
            return img.astype('float32'), self.label
        else:
            img = img.numpy()
            return img.astype('float32'), self.file_list[idx]


train_transform = transforms.Compose([
    transforms.Resize((256, 256)),  # 先调整图片大小至256x256
    transforms.RandomCrop((224, 224)),  # 再随机裁剪到224x224
    transforms.RandomHorizontalFlip(),  # 随机的图像水平翻转，通俗讲就是图像的左右对调
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # 归一化，数值是用ImageNet给出的数值
])

# 把猫的图片和狗的图片分开
cat_files = [tf for tf in train_files if 'cat' in tf]
dog_files = [tf for tf in train_files if 'dog' in tf]

cats = CatDogDataset(cat_files, train_dir, transform=train_transform)  # 猫的数据集类
dogs = CatDogDataset(dog_files, train_dir, transform=train_transform)  # 狗的数据集类

train_set = ConcatDataset([cats, dogs])  # 把猫和狗的数据集合并
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)  # 训练集数据加载器

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

test_set = CatDogDataset(test_files, test_dir, mode='test', transform=test_transform)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

samples, labels = iter(train_loader).next()
plt.figure(figsize=(16, 24))
grid_imgs = torchvision.utils.make_grid(samples[:BATCH_SIZE])
np_grid_imgs = grid_imgs.numpy()
# in tensor, image is (batch, width, height), so you have to transpose it to (width, height, batch) in numpy to show it.
plt.imshow(np.transpose(np_grid_imgs, (1, 2, 0)))


class MineNet(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),  # (224+2*2-11)/4+1=55
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # (55-3)/2+1=27
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),  # (27+2*2-5)/1+1=27
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # (27-3)/2+1=13
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # (13+1*2-3)/1+1=13
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),  # (13+1*2-3)/1+1=13
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),  # 13+1*2-3)/1+1=13
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # (13-3)/2+1=6
        )  # 6*6*128=9126

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(128 * 6 * 6, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
        )
        # softmax
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = self.logsoftmax(x)
        return x


model = MineNet()
# model = MyConvNet().to(DEVICE)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)  # 设置训练细节
scheduler = StepLR(optimizer, step_size=5)  # 设置学习率下降策略
criterion = nn.CrossEntropyLoss()  # 设置损失函数


# model.load_state_dict(torch.load('catdog_mineresnet_10.pth'))


def refreshdataloader():
    cat_files = [tf for tf in train_files if 'cat' in tf]
    dog_files = [tf for tf in train_files if 'dog' in tf]

    val_cat_files = []
    val_dog_files = []

    for i in range(0, 1250):
        r = random.randint(0, len(cat_files) - 1)
        val_cat_files.append(cat_files[r])
        val_dog_files.append(dog_files[r])
        cat_files.remove(cat_files[r])
        dog_files.remove(dog_files[r])

    cats = CatDogDataset(cat_files, train_dir, transform=train_transform)
    dogs = CatDogDataset(dog_files, train_dir, transform=train_transform)

    train_set = ConcatDataset([cats, dogs])
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    val_cats = CatDogDataset(val_cat_files, train_dir, transform=test_transform)
    val_dogs = CatDogDataset(val_dog_files, train_dir, transform=test_transform)

    val_set = ConcatDataset([val_cats, val_dogs])
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    return train_loader, val_loader


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0.0
    train_acc = 0.0
    percent = 10

    for batch_idx, (sample, target) in enumerate(train_loader):
        sample, target = sample.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(sample)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        loss = loss.item()
        train_loss += loss
        pred = output.max(1, keepdim=True)[1]
        train_acc += pred.eq(target.view_as(pred)).sum().item()

        if (batch_idx + 1) % percent == 0:
            print('train epoch: {} [{}/{} ({:.0f}%)]\tloss: {:.6f}\t'.format(
                epoch, (batch_idx + 1) * len(sample), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss))

    train_loss *= BATCH_SIZE
    train_loss /= len(train_loader.dataset)
    train_acc = train_acc / len(train_loader.dataset)
    print('\ntrain epoch: {}\tloss: {:.6f}\taccuracy:{:.4f}% '.format(epoch, train_loss, 100. * train_acc))
    scheduler.step()

    return train_loss, train_acc


def val(model, device, val_loader, epoch):
    model.eval()
    val_loss = 0.0
    correct = 0
    for sample, target in val_loader:
        with torch.no_grad():
            sample, target = sample.to(device), target.to(device)
            output = model(sample)

            val_loss += criterion(output, target).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    val_loss *= BATCH_SIZE
    val_loss /= len(val_loader.dataset)
    val_acc = correct / len(val_loader.dataset)
    print("\nval set: epoch{} average loss: {:.4f}, accuracy: {}/{} ({:.4f}%) \n"
          .format(epoch, val_loss, correct, len(val_loader.dataset), 100. * val_acc))
    return val_loss, 100. * val_acc


def test(model, device, test_loader, epoch):
    model.eval()
    filename_list = []
    pred_list = []
    for sample, filename in test_loader:
        with torch.no_grad():
            sample = sample.to(device)
            output = model(sample)
            pred = torch.argmax(output, dim=1)

            filename_list += [n[:-4] for n in filename]
            pred_list += [p.item() for p in pred]

    print("\ntest epoch: {}\n".format(epoch))

    submission = pd.DataFrame({"id": filename_list, "label": pred_list})
    submission.to_csv('preds_' + str(epoch) + '.csv', index=False)


train_counter = []  # 训练集数量
train_losses = []  # 训练集损失
train_acces = []  # 训练集准确率
val_counter = []  # 验证集数量
val_losses = []  # 验证集损失
val_acces = []  # 验证集准确率

for epoch in range(1, EPOCHS + 1):
    # 刷新读取数据集
    train_loader, val_loader = refreshdataloader()
    # 开始训练并记录训练数据
    tr_loss, tr_acc = train(model, DEVICE, train_loader, optimizer, epoch)
    train_counter.append((epoch - 1) * len(train_loader.dataset))
    train_losses.append(tr_loss)
    train_acces.append(tr_acc)

    # 验证当前训练的预测效果
    vl, va = val(model, DEVICE, val_loader, epoch)
    val_counter.append((epoch - 1) * len(val_loader.dataset))
    val_losses.append(vl)
    val_acces.append(va)

    # 将当前批次模型保存下来
    filename_pth = 'catdog_mineresnet_' + str(epoch) + '.pth'
    torch.save(model.state_dict(), filename_pth)

test(model, DEVICE, test_loader, 1)

fig = plt.figure()  # 创建图像
plt.plot(train_counter, train_losses, color='blue')  # 画出训练损失曲线
plt.scatter(val_counter, val_losses, color='red')  # 画出测试损失散点图
plt.legend(['Train Loss', 'Test Loss'], loc='upper right')  # 图例标识
plt.xlabel('number of training examples seen')  # x轴标签
plt.ylabel('negative log likelihood loss')  # y轴标签


def showDemo():
    examples = enumerate(test_loader)  # 枚举测试集
    batch_idx, (example_data, example_targets) = next(examples)  # 获取第一个批次的数据
    with torch.no_grad():  # 不计算梯度，节省内存
        y_pred = model(example_data)  # 将数据输入网络，得到输出
        _, pred = torch.max(y_pred.data, 1)
    fig = plt.figure()  # 创建图像

    for i in range(6):  # 画出前6个样本的预测结果
        plt.subplot(2, 3, i + 1)  # 2行3列，第 i+1 个子图
        plt.tight_layout()  # 自动适配子图参数，使之填充整个图像区域
        index = random.randint(0, len(example_data) - 1)
        plt.imshow(example_data[index][0], interpolation='none')  # 画出第 i 个样本
        plt.title(
            f"Predict Label: {labelList[pred[index]]}")  # 标题为预测结果
        plt.xticks([])  # 不显示 x 轴刻度
        plt.yticks([])  # 不显示 y 轴刻度
    plt.show()  # 显示图像


showDemo()
