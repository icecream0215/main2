import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        # VGG layers
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, stride=2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, stride=2)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(2, stride=2)
        self.fc1 = nn.Linear(512 * 8 * 8, 4096)
        self.fc2 = nn.Linear(4096, 186)  # 输出维度为186
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        # 确保输入维度正确
        if len(x.shape) == 2:
            x = x.unsqueeze(0)  # 添加batch维度
        x = x.unsqueeze(1)  # 添加通道维度

        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = F.relu(self.conv4(x))
        x = self.pool4(x)
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

if __name__ == '__main__':
    # 测试代码
    model = VGG()
    x = torch.randn(1, 128, 128)  # 示例输入
    output = model(x)
    print(f"Output shape: {output.shape}")  # 应该输出 torch.Size([1, 186])
