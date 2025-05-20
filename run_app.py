"""
启动抑郁症检测系统的简单脚本
"""
import os
import sys

# 确保能找到tcn模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 切换到tcn目录
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "tcn"))

# 导入并运行应用
print("启动抑郁症检测系统...")
from tcn.app import app
import uvicorn

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
