from collections import OrderedDict
import torch.nn as nn
import torch
import numpy as np
import logging
from torch.autograd import Function
from math import sqrt
import torch
import torch.nn.functional as F

class Conv1d(nn.Module): #一维卷积
    def __init__(self) -> None:
        super(Conv1d, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(171, 32, 3, padding=1),  # 卷积：输入通道171，输出32，卷积核大小3，保持尺寸（padding=1）
            nn.BatchNorm1d(32),                # 对32个通道做批归一化
            nn.ReLU(),                         # 激活函数ReLU
            nn.MaxPool1d(2),                   # 最大池化，步长2，降低特征图尺寸
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(32, 64, 3, padding=1),   # 卷积：输入32，输出64
            nn.BatchNorm1d(64),                # 批归一化
            nn.ReLU(),                         # 激活函数
            nn.MaxPool1d(2),                   # 池化
        )
        self.layer3 = nn.Sequential(
            nn.Conv1d(64, 128, 3, padding=1),  # 卷积：输入64，输出128
            nn.BatchNorm1d(128),               # 批归一化
            nn.ReLU(),                        # 激活函数
            nn.MaxPool1d(2),                  # 池化
        )

    def forward(self, input):
        x = self.layer1(input)  # 依次经过第一层卷积、BN、ReLU、池化
        x = self.layer2(x)      # 第二层处理
        x = self.layer3(x)      # 第三层处理
        return x                # 返回最终特征图


class AstroModel(nn.Module):  #
    def __init__(self) -> None:
        super(AstroModel, self).__init__()
        self.conv = nn.Conv1d(128, 256, 1)  # 一维卷积，将128通道映射到256通道，卷积核大小为1，相当于通道变换
        self.dropout = nn.Dropout(0.2)       # Dropout层，丢弃率20%

        self.conv1_1 = nn.Conv1d(128, 128, 3, padding=2, dilation=2)
        self.conv1_2 = nn.Conv1d(128, 128, 3, padding=2, dilation=2)
        # 第一组空洞卷积，使用dilation=2，padding=2以保持尺寸

        self.conv2_1 = nn.Conv1d(128, 128, 3, padding=4, dilation=4)
        self.conv2_2 = nn.Conv1d(128, 128, 3, padding=4, dilation=4)
        # 第二组空洞卷积，dilation=4，padding=4

        self.conv3_1 = nn.Conv1d(128, 128, 3, padding=8, dilation=8)
        self.conv3_2 = nn.Conv1d(128, 128, 3, padding=8, dilation=8)
        # 第三组空洞卷积，dilation=8，padding=8

        self.conv4_1 = nn.Conv1d(128, 256, 3, padding=16, dilation=16)
        self.conv4_2 = nn.Conv1d(256, 256, 3, padding=16, dilation=16)
        # 第四组空洞卷积，第一层将通道由128扩展到256，第二层继续256通道

    def forward(self, x):
        raw = x  # 保存原始输入
        x = F.relu(self.conv1_1(x))  # 第一组第一个空洞卷积 + ReLU激活
        x = self.dropout(x)          # Dropout防止过拟合
        x = self.dropout(self.conv1_2(x))  # 第一组第二个空洞卷积 + Dropout
        raw = F.relu(x + raw)        # 残差连接：原始输入与卷积输出相加后激活

        x = raw                    # 将残差输出作为下一组的输入
        x = F.relu(self.conv2_1(x))  # 第二组第一个卷积 + ReLU
        x = self.dropout(x)
        x = self.dropout(self.conv2_2(x))  # 第二组第二个卷积 + Dropout
        raw = F.relu(x + raw)       # 残差连接

        x = raw                    # 第三组处理
        x = F.relu(self.conv3_1(x))
        x = self.dropout(x)
        x = self.dropout(self.conv3_2(x))
        raw = F.relu(x + raw)       # 残差连接

        x = raw                    # 第四组处理
        x = F.relu(self.conv4_1(x))
        x = self.dropout(x)
        x = self.dropout(self.conv4_2(x))
        raw = self.conv(raw)        # 经过1x1卷积，进行通道扩展
        raw = F.relu(x + raw)       # 最后一次残差连接 + 激活

        return raw                # 返回处理后的特征

        
class TCNModel(nn.Module):
    def __init__(self) -> None:
        super(TCNModel, self).__init__()
        self.Conv1d = Conv1d()          # 使用前面定义的一维卷积模块
        self.AstroModel = AstroModel()  # 使用前面定义的 AstroModel 模块

    def forward(self, input):
        input = input.transpose(1, 2)  # 转置输入张量的维度（从 [batch, seq_len, features] 转换为 [batch, features, seq_len]，以适配Conv1d）
        x = self.Conv1d(input)        # 经过 Conv1d 模块提取特征
        x = self.AstroModel(x)        # 再经过 AstroModel 模块进一步提取特征
        x = x.transpose(1,2)          # 将特征再次转置回来（例如恢复成 [batch, seq_len, features]）
        return x                    # 返回最终的时序特征


class CalculateAttention(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, Qx, Kx, Vx, Qy, Ky, Vy):
        attentionx = torch.matmul(Qx, torch.transpose(Kx, -1, -2))  # 计算 Qx 与 Kx 的点积，得到注意力分数
        attentiony = torch.matmul(Qy, torch.transpose(Ky, -1, -2))  # 同样计算 Qy 与 Ky 的点积
        attention = torch.cat((attentionx, attentiony), dim=1)       # 将两个注意力矩阵在通道维度上拼接
        B, C, H, W = attention.size()  # 获取拼接后张量的尺寸
        attention = attention.reshape(B, 2, C//2, H, W)  # 重塑张量，将两个部分区分开来
        attention = torch.mean(attention, dim=1).squeeze()  # 对第二个维度求均值，得到综合注意力
        attention1 = torch.softmax(attention / sqrt(Qx.size(-1)), dim=-1)  # 对缩放后的注意力进行 softmax
        attention1 = torch.matmul(attention1, Vx)  # 得到加权后的 Vx
        attention2 = torch.softmax(attention / sqrt(Qx.size(-1)), dim=-1)  # 同样对 attention 计算 softmax
        attention2 = torch.matmul(attention2, Vy)  # 得到加权后的 Vy
        return attention1, attention2  # 返回两个注意力输出


class FeedForward(nn.Module):
    def __init__(self, dim_in, hidden_dim, dim_out=None, *, dropout=0.0, f=nn.Conv1d, activation=nn.ELU):
        super(FeedForward, self).__init__()
        dim_out = dim_in if dim_out is None else dim_out

        self.net = nn.Sequential(
            f(in_channels=dim_in, out_channels=hidden_dim, kernel_size=1, padding=0, stride=1),  # 1x1 卷积作为线性映射
            activation(),                                                                     # 激活函数（ELU）
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),                          # Dropout层（如果设置了丢弃率）
            f(in_channels=hidden_dim, out_channels=dim_out, kernel_size=1, padding=0, stride=1), # 第二个1x1卷积
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),                          # Dropout层
        )

    def forward(self, x):
        x = self.net(x)  # 将输入依次经过两层1x1卷积、激活和Dropout
        return x



class Multi_CrossAttention(nn.Module):
    def __init__(self, hidden_size, all_head_size, head_num):
        super().__init__()
        self.hidden_size = hidden_size
        self.all_head_size = all_head_size
        self.num_heads = head_num
        self.h_size = all_head_size // head_num  # 每个头的维度
        assert all_head_size % head_num == 0       # 确保可以均分
        self.linear_q = nn.Linear(hidden_size, all_head_size, bias=False)  # 投影生成查询向量
        self.linear_k = nn.Linear(hidden_size, all_head_size, bias=False)  # 投影生成键向量
        self.linear_v = nn.Linear(hidden_size, all_head_size, bias=False)  # 投影生成值向量
        self.pooling = nn.AdaptiveAvgPool1d(1)    # 自适应平均池化（虽然此处定义了但后续未见调用）
        self.norm = sqrt(all_head_size)             # 缩放因子

    def print(self):
        print(self.hidden_size, self.all_head_size)
        print(self.linear_k, self.linear_q, self.linear_v)

    def forward(self, x, y):
        batch_size = x.size(0)
        # 对 y 进行投影，reshape 成多头形式，并换轴
        q_sx = self.linear_q(y).view(batch_size, -1, self.num_heads, self.h_size).transpose(1, 2)
        # 对 x 进行投影，reshape 成多头形式
        k_sx = self.linear_k(x).view(batch_size, -1, self.num_heads, self.h_size).transpose(1, 2)
        v_sx = self.linear_v(x).view(batch_size, -1, self.num_heads, self.h_size).transpose(1, 2)

        # 反过来，对 x 和 y进行交叉投影
        q_sy = self.linear_q(x).view(batch_size, -1, self.num_heads, self.h_size).transpose(1, 2)
        k_sy = self.linear_k(y).view(batch_size, -1, self.num_heads, self.h_size).transpose(1, 2)
        v_sy = self.linear_v(y).view(batch_size, -1, self.num_heads, self.h_size).transpose(1, 2)

        # 使用 CalculateAttention 计算两组注意力
        attention1, attention2 = CalculateAttention()(q_sx, k_sx, v_sx, q_sy, k_sy, v_sy)
        # 将多头注意力输出合并回原维度，并加上残差
        attention1 = attention1.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.h_size) + x
        attention2 = attention2.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.h_size) + y

        return attention1, attention2


class ConvNet1d(nn.Module):
    def __init__(self) -> None:
        super(ConvNet1d, self).__init__()
        self.fc = nn.Linear(256, 128)  # 全连接层，将输入维度256映射到128

    def forward(self, input):
        sizeTmp = input.size(1)  # 记录输入的时间步数或序列长度
        batch_size = input.size(0)
        # 将输入展平（将批次和时间步合并），便于全连接层处理
        outConv1d = input.contiguous().view(input.size(0) * input.size(1), -1)
        output = self.fc(outConv1d)  # 经过全连接层
        output = output.view(batch_size, sizeTmp, -1)  # 重新恢复成 [batch, time_step, features] 的形状

        return output




class gateRegress(): #
    def __init__(self) -> None:
        pass
    def forward(self, ):
        pass

class Regress2(nn.Module):
    def __init__(self) -> None:
        super(Regress2, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(186, 64),  # 将186维特征映射到64维
            nn.ELU(),            # ELU 激活函数
            nn.Dropout(p=0.1),     # Dropout，丢弃率10%
            nn.Linear(64, 2),     # 将64维映射到2维（输出2个数值，可能代表两个类别或回归值）
            nn.ELU()             # 激活函数
        )

    def forward(self, x):
        x = x.view(-1, 186)  # 将输入reshape为二维张量，每行186维
        x = self.fc(x)       # 依次经过全连接层、激活和Dropout
        return x



class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.TCNModel = TCNModel()         # 用于处理视频特征的时序卷积网络
        self.Conv1dModel = ConvNet1d()       # 对视频特征进行全连接变换
        self.Regress = Regress2()            # 最后的回归或分类模块

        self.softmax = torch.nn.Softmax(dim=1)  # softmax激活，用于输出概率分布
        self.conv = nn.Conv1d(in_channels=114, out_channels=186, kernel_size=1, padding=0, stride=1)
        # 1x1卷积，用于调整特征通道，从114到186

        self.mhca = Multi_CrossAttention(hidden_size=128, all_head_size=128, head_num=4)
        # 多头交叉注意力模块，参数设定表示将128维特征分成4个头，每个头32维（128/4）

        self.norm = nn.LayerNorm(128*2)   # LayerNorm正则化层，对拼接后特征（维度128*2）进行归一化
        self.FFN = FeedForward(dim_in=186, hidden_dim=186*2, dim_out=186)
        # 前馈网络，用于对特征进行非线性变换，保持输入输出维度为186，中间扩展到372

        self.norm2 = nn.LayerNorm(128*2)   # 第二个LayerNorm层
        self.pooling = nn.AdaptiveAvgPool1d(1)  # 自适应平均池化，将时间步维度降为1

    def forward(self, inputVideo, inputAudio):
        # inputVideo: 视频输入特征
        # inputAudio: 音频输入特征

        inputVideo = self.TCNModel(inputVideo)
        # 将视频输入通过 TCN 模型提取时序特征

        outputConv1dVideo = self.Conv1dModel(inputVideo)
        # 对视频特征经过全连接变换，调整特征维度

        outputConv1dVideo = self.conv(outputConv1dVideo)
        # 使用1x1卷积调整视频特征通道数，从114变成186

        output1, output2 = self.mhca(outputConv1dVideo, inputAudio)
        # 利用多头交叉注意力模块，将视频特征和音频特征进行交互，分别输出两组注意力融合结果

        outputFeature = torch.cat((output1, output2), dim=2)
        # 将两组特征在特征维度上拼接，形成融合后的特征

        outputFeature = self.FFN(self.norm(outputFeature)) + outputFeature
        # 先对拼接特征做LayerNorm归一化，然后通过前馈网络FFN处理，并与原特征做残差相加

        output = self.norm2(outputFeature)
        # 再次归一化

        output = self.pooling(output).reshape(output.shape[0], -1)
        # 自适应池化降采样（将时序维度降为1），并reshape为二维张量，便于后续全连接处理

        result = self.Regress(output)
        # 通过回归模块（或分类模块）生成最终结果

        result = result.squeeze(-1)
        result = self.softmax(result)
        # 对输出进行squeeze和softmax，最终输出概率分布

        return result


if __name__ == '__main__':
    model = Net().cuda()  # 实例化 Net 模型，并将其转移到 GPU 上
    Conv1dModel = ConvNet1d()  # 单独实例化一个 ConvNet1d 模型（这里实例化后未见使用）
    x1 = torch.randn(4, 186, 128).cuda()  # 随机生成一个张量作为音频输入，形状：[batch=4, time_steps=186, features=128]
    x2 = torch.randn(4, 915, 171).cuda()  # 随机生成一个张量作为视频输入，形状：[batch=4, time_steps=915, features=171]
    y = model(x2, x1)  # 前向传播，输入视频和音频数据
    print(y.shape)     # 打印输出结果的形状

    

