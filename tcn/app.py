import io
import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

# -----------------------------
# 1. 导入并初始化你的模型
from tcnmodel import Net  # 请确保你的模型类 Net 定义正确

device = torch.device("cpu")
model = Net()

# 加载模型权重（提取 checkpoint 中的 state_dict）
checkpoint = torch.load(r"C:\Users\ice-creme\Desktop\main\tcnfeature\model.pth", map_location=device, weights_only=False)
state_dict = checkpoint["net"]  # 提取模型的 state_dict
model.load_state_dict(state_dict)
model.to(device)
model.eval()

# 模型期望的输入尺寸（根据你的测试代码）
NORMAL_VIDEO_SHAPE = 915   # 视频序列长度
DESIRED_VIDEO_WIDTH = 171  # 每帧视频特征数
NORMAL_AUDIO_SHAPE = 186   # 音频序列长度
DESIRED_AUDIO_WIDTH = 128  # 音频特征数

# 定义辅助函数：对输入数组进行裁剪/零填充
def pad_or_crop(array, target_length, axis=0):
    current = array.shape[axis]
    if current > target_length:
        slicer = [slice(None)] * array.ndim
        slicer[axis] = slice(0, target_length)
        return array[tuple(slicer)]
    elif current < target_length:
        pad_width = [(0, 0)] * array.ndim
        pad_width[axis] = (0, target_length - current)
        return np.pad(array, pad_width, mode="constant")
    else:
        return array

# -----------------------------
# 2. 构建 FastAPI 接口
app = FastAPI(title="PyTorch Model Inference API")

# 定义一个首页 GET 接口，返回 HTML 表单页面
@app.get("/", response_class=HTMLResponse)
async def index():
    html_content = """
    <html>
        <head>
            <title>上传 npy 文件进行推理</title>
        </head>
        <body>
            <h1>上传 npy 文件</h1>
            <form action="/predict" method="post" enctype="multipart/form-data">
                <div>
                    <label for="video_file">视频 npy 文件:</label>
                    <input type="file" name="video_file" accept=".npy" required>
                </div>
                <br>
                <div>
                    <label for="audio_file">音频 npy 文件:</label>
                    <input type="file" name="audio_file" accept=".npy" required>
                </div>
                <br>
                <div>
                    <input type="submit" value="提交">
                </div>
            </form>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# 定义 /predict POST 接口，处理文件上传和模型推理
@app.post("/predict")
async def predict(video_file: UploadFile = File(...), audio_file: UploadFile = File(...)):
    try:
        # 读取上传的 .npy 文件内容
        video_bytes = await video_file.read()
        audio_bytes = await audio_file.read()

        # 用 BytesIO 将字节流转为 npy 数组
        video_array = np.load(io.BytesIO(video_bytes))
        audio_array = np.load(io.BytesIO(audio_bytes))

        # 对视频和音频数据进行裁剪或零填充
        video_array = pad_or_crop(video_array, NORMAL_VIDEO_SHAPE, axis=0)
        audio_array = pad_or_crop(audio_array, NORMAL_AUDIO_SHAPE, axis=0)

        # 确保第二维的特征数量符合要求
        video_array = pad_or_crop(video_array, DESIRED_VIDEO_WIDTH, axis=1)
        audio_array = pad_or_crop(audio_array, DESIRED_AUDIO_WIDTH, axis=1)

        # 转换为 torch Tensor，并添加 batch 维度
        video_tensor = torch.tensor(video_array, dtype=torch.float32).unsqueeze(0).to(device)
        audio_tensor = torch.tensor(audio_array, dtype=torch.float32).unsqueeze(0).to(device)

        # 模型推理
        with torch.no_grad():
            output = model(video_tensor, audio_tensor)  # 期望输出 shape: (1, 2)

        # 处理输出：squeeze 去掉 batch 维度
        probs = output.squeeze(0).cpu().numpy()  # shape: (2,)
        percentages = probs * 100  # 转换为百分比

        # 预测类别：取概率最大的类别索引
        predicted_idx = int(np.argmax(percentages))
        label_map = {0: "Class 0", 1: "Class 1"}

        return {
            "predicted_class": label_map[predicted_idx],
            "probabilities": {
                label_map[0]: f"{percentages[0]:.2f}%",
                label_map[1]: f"{percentages[1]:.2f}%"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# -----------------------------
# 3. 运行服务
if __name__ == '__main__':
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
