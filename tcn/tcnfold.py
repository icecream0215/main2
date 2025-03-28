import pandas as pd
from sklearn.model_selection import StratifiedKFold
import torch
import logging
import os
import time
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.optim.lr_scheduler import LambdaLR
from torch import nn
from tcnmodel import Net  # 引入模型
from kfoldLoader import MyDataLoader  # 引入数据加载器
from tqdm import tqdm

# 配置日志
tim = time.strftime('%m_%d__%H_%M', time.localtime())
filepath = '/root/autodl-tmp/model/logs' + str(tim)
savePath = "/root/autodl-tmp/model/checkpoints" + str(tim)

if not os.path.exists(filepath):
    os.makedirs(filepath)
if not os.path.exists(savePath):
    os.makedirs(savePath)

logging.basicConfig(level=logging.NOTSET,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filename=filepath + '/' + 'training_log.log',
                    filemode='w')

# 模型训练和评估
def train_and_evaluate(VideoPath, AudioPath, X_train, X_test, labelPath, numkfold, epoch_size=300, lr=0.00001):
    mytop = 0
    topacc = 60
    ps, rs, f1s = [], [], []
    
    # 初始化数据加载器
    trainSet = MyDataLoader(VideoPath, AudioPath, X_train, labelPath, "train")
    trainLoader = DataLoader(trainSet, batch_size=15, shuffle=True)  #训练器的批次大小是15
    devSet = MyDataLoader(VideoPath, AudioPath, X_test, labelPath, "dev")
    devLoader = DataLoader(devSet, batch_size=4, shuffle=False) #测试的批次大小是4
    
    logging.info(f"Training started for fold {numkfold}")  #写？日志

    # 创建模型
    model = Net().cuda()  # 使用tcnmodel中的模型
    lossFunc = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False)

    # 计算训练步数
    train_steps = len(trainLoader) * epoch_size
    warmup_steps = 0
    target_steps = len(trainLoader) * epoch_size

    # 学习率调度器
    scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 0.5 * (1 + np.cos(np.pi * epoch / epoch_size)))  #余弦调整器
    
    # 训练过程
    for epoch in range(1, epoch_size + 1):
        model.train()
        total = 0
        correct = 0
        traloss_one = 0
        loop = tqdm(enumerate(trainLoader), total=len(trainLoader))
        
        for batch_idx, (videoData, audioData, label) in loop:
            if torch.cuda.is_available():
                videoData, audioData, label = videoData.cuda(), audioData.cuda(), label.cuda()
            
            output = model(videoData, audioData)
            traLoss = lossFunc(output, label.long())
            traloss_one += traLoss
            
            optimizer.zero_grad()
            traLoss.backward()
            optimizer.step()
            scheduler.step()
            
            _, predicted = torch.max(output.data, 1)
            total += label.size(0)
            correct += predicted.eq(label.data).cpu().sum()
            
            loop.set_description(f'Train Epoch [{epoch}/{epoch_size}]')
            loop.set_postfix(loss=traloss_one / (batch_idx + 1))
        
        logging.info(f'Epoch {epoch}, Train Loss: {traloss_one/len(trainLoader)}, Train Acc: {100.0*correct/total}%')  #写？日志

        # 验证阶段
        if epoch % 10 == 0:
            model.eval()
            correct = 0
            total = 0
            label2 = []
            pre2 = []
            with torch.no_grad():
                for batch_idx, (videoData, audioData, label) in enumerate(devLoader):
                    if torch.cuda.is_available():
                        videoData, audioData, label = videoData.cuda(), audioData.cuda(), label.cuda()
                    devOutput = model(videoData, audioData)
                    _, predicted = torch.max(devOutput.data, 1)
                    total += label.size(0)
                    correct += predicted.eq(label.data).cpu().sum()

                    label2.append(label.data.cpu())
                    pre2.append(predicted.cpu())

                # 将标签和预测结果拼接成一个连续的张量
                label2 = torch.cat(label2, dim=0).numpy()  # 展平并转换为 numpy 数组
                pre2 = torch.cat(pre2, dim=0).numpy()  # 展平并转换为 numpy 数组
                
                acc = 100.0 * correct / total
                p = precision_score(label2, pre2, average='weighted')
                r = recall_score(label2, pre2, average='weighted')
                f1score = f1_score(label2, pre2, average='weighted')

                
                logging.info(f'Epoch {epoch}, Validation Acc: {acc}%, Precision: {p}, Recall: {r}, F1: {f1score}')
                
                if acc > mytop:
                    mytop = acc
                    ps.append(p)
                    rs.append(r)
                    f1s.append(f1score)

                if acc > topacc:
                    topacc = acc
                    checkpoint = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch, 'scheduler': scheduler.state_dict()}  #我pth文件中的东西
                    torch.save(checkpoint, os.path.join(savePath, f"model_fold_{numkfold}_epoch_{epoch}_acc_{acc}.pth"))  #保存当前最好的模型到检查点
                    
    logging.info(f"Top Acc: {mytop}")
    logging.info(f"Average Precision: {sum(ps)/len(ps)}")
    logging.info(f"Average Recall: {sum(rs)/len(rs)}")
    logging.info(f"Average F1: {sum(f1s)/len(f1s)}")
    return model

if __name__ == "__main__":
    # 输入特征及标签的路径
    VideoPath = "/root/autodl-tmp/Video_tcn"  # 视频特征路径
    AudioPath = "/root/autodl-tmp/Audio_feature"  # 音频特征路径
    labelPath = "/root/LMVD-main/label/label"  # 标签文件路径
    
    # 获取视频文件名列表
    X = np.array(os.listdir(VideoPath))  # 视频特征的文件名列表，创建np数组来储存它
    
    # 初始化标签列表
    Y = []

    # 加载标签数据
    for filename in X:
        # 标签文件和视频文件具有相同的前缀，标签文件为 CSV 格式
        label_file = os.path.join(labelPath, f"{filename.split('.')[0]}_Depression.csv")
        
        # 读取标签文件，每个文件只有一个标签，且标签在第一列
        if os.path.exists(label_file):  # 确保标签文件存在
            label_data = pd.read_csv(label_file)
            
            # 标签在文件的第一列，0 或 1
            label = int(label_data.columns[0])  # 获取标签
            
            # 将标签添加到 Y 列表
            Y.append(label)
        else:
            print(f"Warning: Label file {label_file} does not exist!")

    Y = np.array(Y)  # 转换为 NumPy 数组

    # 初始化 k-fold 交叉验证
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
    numkfold = 0
    for train_index, test_index in kf.split(X, Y):  #这个循环是怎么个事?
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        
        # 输出当前训练折数
        numkfold += 1
        logging.info(f"Training fold {numkfold}")
        
        # 调用训练和评估函数
        train_and_evaluate(VideoPath, AudioPath, X_train, X_test, labelPath, numkfold)

    logging.info("Training complete")