import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader

from model import *

# 定义测试的设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 准备数据集
test_data = torchvision.datasets.CIFAR10(root="./data",train=False,transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data_size = len(test_data)
print("测试集的长度为:{}".format(test_data_size))
test_dataloader = DataLoader(test_data,batch_size=64)

model = torch.load("./model/yuhui_999.pth",weights_only=False)
model.eval()
total_test_loss = 0
total_accuracy = 0
with torch.no_grad():
    for data in test_dataloader:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = model(imgs)
        accuracy = (outputs.argmax(1) == targets).sum()
        total_accuracy = total_accuracy + accuracy
print("整体测试集上的正确率:{}".format(total_accuracy/test_data_size))