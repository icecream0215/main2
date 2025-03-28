import pandas as pd  # 导入 pandas，用于数据读取和处理
import os  # 导入 os 模块，用于文件和目录操作
import numpy as np  # 导入 numpy，用于数值计算和数组操作

# 定义 validFrame 函数，用于处理帧数据，使得无效帧（帧中第5个元素为0）的后续部分被替换为最近有效帧的对应值
def validFrame(frames):
    # 遍历所有帧，找到最后一个有效帧（第5个元素为1）的数据，并保存在 validFrame 变量中
    for row in range(frames.shape[0]):
        if frames[row][4] == 1:
            validFrame = frames[row]

    # 再次遍历所有帧，遇到无效帧（第5个元素为0）时，将该帧从第6个元素开始的数据替换为最近一次有效帧的数据
    for row in range(frames.shape[0]):
        if frames[row][4] == 0:
            frames[row][5:] = validFrame[5:]
        if frames[row][4] == 1:
            validFrame = frames[row]  # 更新最近一次有效帧

    return frames  # 返回处理后的帧数据


# 定义 chouzhen 函数，用于从 _feature 列表中每隔6个元素抽取一次，组合成新的数组
def chouzhen(_feature):
    flag = 0
    for i in range(0, len(_feature), 6):  # 每隔6个取一次数据
        if flag == 0:
            feature = _feature[i]  # 第一次取出的数据直接赋值给 feature
            flag = 1
        else:
            feature = np.vstack((feature, _feature[i]))  # 之后的每个数据垂直堆叠到 feature 数组中
    return feature  # 返回抽取后的特征数组


# 定义 split 函数，用于对数据进行切分和填充
def split(data):
    # 取 data 数组前 5490 行，然后调用 chouzhen 函数对其进行抽样
    _data = chouzhen(data[:5490, ])

    # 如果抽样后的数据行数小于915，则用零值数组进行填充，使得行数达到915
    if _data.shape[0] < 915:
        zeros = np.zeros([(915 - _data.shape[0]), _data.shape[1]])
        _data = np.vstack((_data, zeros))
    return _data  # 返回处理后的数据


# 定义 getTCNVideoFeature 函数，用于从视频特征的 CSV 文件中提取特征并保存为 numpy 文件
def getTCNVideoFeature(trainPath, targetPath):
    files = os.listdir(trainPath)  # 获取 trainPath 目录下的所有文件
    for file in files:
        if file.endswith(".csv"):  # 只处理 CSV 文件
            # 读取 CSV 文件，并转换为 numpy 数组
            file_csv = pd.read_csv(os.path.join(trainPath, file))
            data = np.array(file_csv)
            try:
                # 调用 validFrame 函数处理帧数据（补全无效帧信息）
                data = validFrame(data)
            except:
                print('Video issues', file)  # 如果处理出现异常，打印出问题文件并跳过
                continue

            # 调用 split 函数对数据进行抽样和填充
            data = split(data)

            # 删除 data 数组中的前5列（列索引 0 到 4），这些列数据不需要
            data = np.delete(data, [0, 1, 2, 3, 4], axis=1)

            # 分别提取不同区域的特征：
            # 提取 gaze 特征（前6列）
            gaze = data[:, 0:6]
            # 创建与 gaze 同样形状的全零数组
            gaze_zero = np.zeros_like(gaze)
            # 将 gaze 与全零数组水平拼接，得到更宽的 gaze 特征
            gaze = np.hstack((gaze, gaze_zero))
            # 提取 pose 特征（第288到293列）
            pose = data[:, 288:294]
            # 提取 features 特征（第294到429列）
            features = data[:, 294:430]
            # 提取 au 特征（第430到446列）
            au = data[:, 430:447]

            # # 删除 au 特征中第6列（索引为5）的数据
            # au = np.delete(au, [5], axis=1)
            # # 交换 au 特征中列索引 12 和 13 的数据
            # au[:, [12, 13]] = au[:, [13, 12]]
            # # 再交换 au 特征中列索引 13 和 14 的数据
            # au[:, [13, 14]] = au[:, [14, 13]]

            # 选择 au 特征作为基础特征
            feature = au
            # 将 features、gaze 和 pose 特征与 au 特征水平拼接，形成最终的特征矩阵
            feature = np.hstack((feature, features, gaze, pose))

            try:
                # 断言检查：确保 feature 中没有 NaN 值
                assert np.isnan(feature).sum() == 0, print(file)
            except:
                print('There is a null value present：', file)  # 如果存在空值，则打印提示信息

            # 将处理后的特征矩阵保存为 .npy 文件，文件名与 CSV 文件名相同（去掉扩展名）
            np.save(os.path.join(targetPath, file.split(".")[0]), feature)


# 主程序入口
if __name__ == "__main__":
    # 调用 getTCNVideoFeature 函数，传入视频特征 CSV 文件所在的目录和保存处理后特征的目标目录
    # 注意：使用时需要将 "Video feature path" 和 "save tcnfeature path" 修改为实际的文件夹路径
    getTCNVideoFeature(r"C:\Users\ice-creme\Desktop\main\npy", r"C:\Users\ice-creme\Desktop\main\npy")
