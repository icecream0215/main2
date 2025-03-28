import numpy as np
import csv

def npy_to_csv(npy_file, csv_file):
    # 加载npy文件
    data = np.load(npy_file)

    # 确保数据是二维的
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    # 打印列数
    num_columns = data.shape[1]
    print(f"该npy文件有 {num_columns} 列")

    # 将数据写入CSV文件
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)

    print(f"转换完成: {csv_file}")

# 示例用法
npy_file = r"C:\Users\ice-creme\Desktop\main\npy\范例数据\001.npy"  # 替换为你的.npy文件路径
csv_file = "001.csv"  # 你希望保存的.csv文件路径
npy_to_csv(npy_file, csv_file)
