# 基于视听多模态的抑郁症检测系统

这是一个使用视听多模态深度学习模型进行抑郁症检测的系统，通过分析用户上传的视频中的面部表情、语音特征等信息，预测潜在的抑郁风险。

## 功能特性

- 用户认证系统（支持患者、医生和管理员角色）
- 视频上传和分析
- 抑郁风险评估和分类
- 历史记录查询和统计
- 数据分析和可视化
- 用户管理

## 技术栈

- Python 3.9+
- FastAPI 框架
- PyTorch 深度学习
- SQLite 数据库
- OpenFace 面部特征提取
- 前端: HTML, CSS, JavaScript, Chart.js

## 安装指南

### 1. 克隆仓库

```bash
git clone <repository-url>
cd submodel/main2
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 安装 OpenFace

本系统依赖于 OpenFace 进行面部特征提取。请从 [OpenFace官网](https://github.com/TadasBaltrusaitis/OpenFace) 下载并安装 OpenFace。

安装完成后，有两种方式配置 OpenFace 路径:

- 设置环境变量:
  ```
  # Windows
  set OPENFACE_DIR=C:\path\to\OpenFace_2.2.0_win_x64
  
  # Linux/Mac
  export OPENFACE_DIR=/path/to/OpenFace
  ```

- 或者将 OpenFace 安装在上级目录 `../OpenFace_2.2.0_win_x64` 中

### 4. 运行应用

```bash
cd tcn
python app.py
```

服务器将在 http://localhost:8000 启动。

## 使用指南

### 用户角色说明

- **患者**: 可以上传视频进行分析，查看自己的历史记录
- **医生**: 可以上传视频为患者进行分析，查看患者的历史记录和统计数据
- **管理员**: 拥有全部功能权限，可管理用户、查看系统统计数据

### 主要功能

1. 注册/登录系统
2. 上传视频文件进行分析
3. 查看分析结果和历史记录
4. 管理员可查看统计数据和管理用户

## 模型说明

本系统使用的是TCN (Temporal Convolutional Network) 和 VGG 网络的组合模型：

- VGG 网络用于提取音频特征
- TCN 网络用于处理时序特征
- 面部特征通过 OpenFace 提取

## 注意事项

- 视频文件大小限制为200MB
- 支持的视频格式: MP4, AVI, MOV
- 需要互联网连接以加载前端资源
- 初始用户可通过注册创建，管理员账户需要手动在数据库中创建

## 许可证

[MIT License](LICENSE)
